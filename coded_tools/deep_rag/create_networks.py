
# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san SDK Software in commercial settings.
#
# END COPYRIGHT

from typing import Any
from typing import Dict
from typing import List

import aiofiles

# from asyncio import Event
from copy import deepcopy
from json import dumps
from logging import getLogger
from logging import Logger
from pathlib import Path

from leaf_common.config.file_of_class import FileOfClass
# from leaf_common.parsers.dictionary_extractor import DictionaryExtractor
from leaf_common.persistence.easy.easy_hocon_persistence import EasyHoconPersistence

from neuro_san.interfaces.coded_tool import CodedTool
from neuro_san.interfaces.reservation import Reservation
from neuro_san.interfaces.reservationist import Reservationist

from neuro_san.internals.graph.filters.string_common_defs_config_filter import StringCommonDefsConfigFilter


class CreateNetworks(CodedTool):
    """
    CodedTool implementation that creates a single agent network that processes
    a deep_rag grouping of groups.  This can include the front-man for the entire
    deep_rag tree.

    For the time being we are doing somewhat of a fixed hierarchy:
        * front man -> groups (1 network)
        * each group -> all content files (1 network for each)

    We would need to get fancier with the agent network that feeds this tool
    in order to have multiple layers of groups. Not there yet.
    """

    TEMPLATE_FRONT_MAN_INDEX: int = 0

    def __init__(self):
        """
        Constructor
        """
        # Only want to do these things once.
        file_of_class = FileOfClass(__file__)
        template_file: str = file_of_class.get_file_in_basis("group_template.hocon")
        persistence = EasyHoconPersistence()
        self.network_template: Dict[str, Any] = persistence.restore(file_reference=template_file)

        self.logger: Logger = getLogger(self.__class__.__name__)

        # Stuff that gets filled in by args upon ainvoke() call
        self.grouping_json: Dict[str, Any] = {}
        self.files_directory: str = None

        # Stuff that gets constructed which is commonly accessible
        self.name_to_network: Dict[str, str] = {}

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
        """
        Called when the coded tool is invoked asynchronously by the agent hierarchy.
        Strongly consider overriding this method instead of the "easier" synchronous
        invoke() version above when the possibility of making any kind of call that could block
        (like sleep() or a socket read/write out to a web service) is within the
        scope of your CodedTool and can be done asynchronously, especially within
        the context of your CodedTool running within a server.

        If you find your CodedTools can't help but synchronously block,
        strongly consider looking into using the asyncio.to_thread() function
        to not block the EventLoop for other requests.
        See: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
        Example:
            async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> Any:
                return await asyncio.to_thread(self.invoke, args, sly_data)

        :param args: An argument dictionary whose keys are the parameters
                to the coded tool and whose values are the values passed for them
                by the calling agent.  This dictionary is to be treated as read-only.
        :param sly_data: A dictionary whose keys are defined by the agent hierarchy,
                but whose values are meant to be kept out of the chat stream.

                This dictionary is largely to be treated as read-only.
                It is possible to add key/value pairs to this dict that do not
                yet exist as a bulletin board, as long as the responsibility
                for which coded_tool publishes new entries is well understood
                by the agent chain implementation and the coded_tool implementation
                adding the data is not invoke()-ed more than once.
        :return: A return value that goes into the chat stream.
        """
        # Get most args as members
        empty: Dict[str, Any] = {}
        self.grouping_json: Dict[str, Any] = args.get("grouping_json", empty)
        self.files_directory: str = args.get("files_directory")
        reservationist: Reservationist = args.get("reservationist")

        logstr: str = dumps(self.grouping_json, indent=4, sort_keys=True)
        self.logger.info("grouping_json is %s", logstr)

        # Get the list of the groups
        groups: List[Dict[str, Any]] = self.grouping_json.get("groups")

        # Make a dictionary of name -> group
        name_to_group: Dict[str, Dict[str, Any]] = {}
        for group in groups:
            name: str = group.get("name")
            name_to_group[name] = group

        name_to_network: Dict[str, Dict[str, Any]] = await self.make_leaf_networks(name_to_group)
        name_to_reservation = await self.reserve_leaf_networks(reservationist, name_to_network)

        group_network: Dict[str, Any] = self.make_group_network(name_to_reservation.keys())

        return group_network

    async def make_leaf_networks(self, name_to_group: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Assumes each group is a leaf spec.
        No groups container groups yet.
        """

        # Make a dictionary of name -> network name as we create the leaf networks
        group_name_to_network: Dict[str, Dict[str, Any]] = {}
        for group_name, group in name_to_group:

            # If the group has files, then it's a leaf network
            if group.get("files"):
                network: Dict[str, Any] = await self.create_one_leaf_network(group)
                group_name_to_network[group_name] = network

        return group_name_to_network

    async def create_one_leaf_network(self, group: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create an agent network spec for a single leaf group, given the group description.
        """

        agent_spec: Dict[str, Any] = deepcopy(self.network_template)
        tools: List[Dict[str, Any]] = agent_spec.get("tools")

        # The last item in the tools list of the template is the template for a content node.
        content_template: Dict[str, Any] = tools.pop()
        files: Dict[str, str] = group.get("files")

        # Create each content-focused node
        content_tools: List[str] = []
        for file_name, tool_name in files:

            content_agent: Dict[str, Any] = await self.create_one_content_agent(file_name, tool_name, content_template)

            # Add to list of tool specs for network
            tools.append(content_agent)

            # Add to list of tools for front man
            content_tools.append(tool_name)

        # Start out with the front man from the template, but replace him with what's made.
        front_man: Dict[str, Any] = tools[self.TEMPLATE_FRONT_MAN_INDEX]
        tools[self.TEMPLATE_FRONT_MAN_INDEX] = self.create_front_man(front_man, group, content_tools)

        return agent_spec

    async def create_one_content_agent(self, file_name: str, tool_name: str, content_template: Dict[str, Any]) \
            -> Dict[str, Any]:
        """
        Creates a single agent node that sponsors one section of the content
        """

        # Asynchronously read the content of the file
        filepath = Path(self.files_directory) / file_name
        async with aiofiles.open(filepath, "r") as my_file:
            file_content: str = my_file.read()

        # Create the content agent spec by replacing strings in strategic places
        content_agent: Dict[str, Any] = deepcopy(content_template)
        replacements: Dict[str, Any] = {
            "one_content_file": tool_name,
            "content": file_content,
        }
        string_filter = StringCommonDefsConfigFilter(replacements)
        content_agent = string_filter.filter(content_agent)

        return content_agent

    def create_front_man(self, front_man: Dict[str, Any],
                         group: Dict[str, Any],
                         tools: List[str] = None) -> Dict[str, Any]:
        """
        Creates a front man
        """

        # Replace strings in the front man first
        replacements: Dict[str, Any] = {
            "one_group": group.get("name"),
            "group_description": group.get("description"),
            "structure_description": self.grouping_json.get("description")
        }
        string_filter = StringCommonDefsConfigFilter(replacements)
        front_man = string_filter.filter(front_man)
        front_man["tools"] = tools

        return front_man

    async def reserve_leaf_networks(self, reservationist: Reservationist,
                                    name_to_network: Dict[str, Dict[str, Any]]) \
            -> Dict[str, Reservation]:
        """
        Creates reservations for each named network
        """
        name_to_reservation: Dict[str, Reservation] = {}
        return name_to_reservation

    def make_group_network(self, external_tools: List[str]) -> Dict[str, Any]:

        agent_spec: Dict[str, Any] = deepcopy(self.network_template)
        tools: List[Dict[str, Any]] = agent_spec.get("tools")

        # We don't need the content node, we are using external networks for those.
        _ = tools.pop()

        # Start out with the front man from the template, but replace him with what's made.
        front_man: Dict[str, Any] = tools[self.TEMPLATE_FRONT_MAN_INDEX]
        tools[self.TEMPLATE_FRONT_MAN_INDEX] = self.create_front_man(front_man, self.grouping_json, external_tools)

        return agent_spec
