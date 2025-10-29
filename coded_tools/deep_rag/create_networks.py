
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

from asyncio import Event
from copy import deepcopy
from json import dumps
from logging import getLogger
from logging import Logger
from pathlib import Path

from leaf_common.config.file_of_class import FileOfClass
from leaf_common.persistence.easy.easy_hocon_persistence import EasyHoconPersistence

from neuro_san.interfaces.coded_tool import CodedTool
from neuro_san.internals.graph.filters.string_common_defs_config_filter import StringCommonDefsConfigFilter
from neuro_san.internals.graph.filters.dictionary_common_defs_config_filter import DictionaryCommonDefsConfigFilter
from neuro_san.interfaces.reservation import Reservation
from neuro_san.interfaces.reservationist import Reservationist


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
    ONE_HOUR: float = 60 * 60
    LIFETIME: float = ONE_HOUR

    def __init__(self):
        """
        Constructor
        """
        # Only want to do these things once.
        persistence = EasyHoconPersistence()
        file_of_class = FileOfClass(__file__)
        template_file: str = file_of_class.get_file_in_basis("group_template.hocon")
        self.network_template: Dict[str, Any] = persistence.restore(file_reference=template_file)

        aaosa_file: str = file_of_class.get_file_in_basis("../../registries/aaosa_basic.hocon")
        self.aaosa_defs: Dict[str, Any] = persistence.restore(file_reference=aaosa_file)

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

        logstr: str = dumps(self.grouping_json, indent=4, sort_keys=True)
        self.logger.info("grouping_json is %s", logstr)

        reservationist: Reservationist = args.get("reservationist")
        deployments: Dict[Reservation, Dict[str, Any]] = await self.assemble_deployments(reservationist)

        # Deploy the reservations with confirmation event
        # If you don't really need to wait until the new agent(s) has been deployed
        # then set confirmation=False, and don't bother about waiting for the Event.
        deployed_event: Event = None
        try:
            async with reservationist:
                deployed_event = await reservationist.deploy(deployments, confirmation=True)

        except ValueError as exception:
            # Report exceptions from below as errors here.
            error: str = f"{exception}"
            self.logger.error(error)
            return error

        if deployed_event is not None:
            await deployed_event.wait()

        # Assemble the output
        reservation_info: List[Dict[str, Any]] = self.assemble_reservation_info(deployments.keys())
        sly_data["agent_reservations"] = reservation_info
        sly_data["aa_grouping_json"] = self.grouping_json

        entry: Dict[str, Any] = reservation_info[-1]
        entry_reservation_id: str = entry.get("reservation_id")
        entry_lifetime: str = entry.get("lifetime_in_seconds")
        output: str = f"The main agent to access your deep rag network is {entry_reservation_id}" + \
                      f"Hurry, it's only available for {entry_lifetime} seconds."
        return output

    async def assemble_deployments(self, reservationist: Reservationist) -> Dict[Reservation, Dict[str, Any]]:
        """
        Create all the networks that are to be deployed together.
        """

        # Get the list of the groups
        groups: List[Dict[str, Any]] = self.grouping_json.get("groups")

        # Make a dictionary of name -> group
        name_to_group: Dict[str, Dict[str, Any]] = {}
        for group in groups:
            name: str = group.get("name")
            name_to_group[name] = group

        # Create the leaf networks and make Reservations for them
        name_to_network: Dict[str, Dict[str, Any]] = await self.make_leaf_networks(name_to_group)
        deployments: Dict[Reservation, Dict[str, Any]] = await self.reserve_leaf_networks(reservationist,
                                                                                          name_to_network)

        # Use the reservations as tools in the top-level group network
        group_network: Dict[str, Any] = self.make_group_network(deployments.keys())
        filtered_name: str = self.grouping_json.get("name")
        filtered_name = filtered_name.replace(" ", "_")
        reservation: Reservation = await reservationist.reserve(lifetime_in_seconds=self.LIFETIME,
                                                                prefix=filtered_name)
        deployments[reservation] = group_network

        return deployments

    async def make_leaf_networks(self, name_to_group: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Assumes each group is a leaf spec.
        No groups container groups yet.
        """

        # Make a dictionary of name -> network name as we create the leaf networks
        group_name_to_network: Dict[str, Dict[str, Any]] = {}
        for group_name, group in name_to_group.items():

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

        logstr: str = dumps(group, indent=4, sort_keys=True)
        self.logger.info("Processing group: %s", logstr)

        files: Dict[str, str] = group.get("files")

        # Create each content-focused node
        content_tools: List[str] = []
        for file_name, tool_name in files.items():

            content_agent: Dict[str, Any] = await self.create_one_content_agent(file_name, tool_name, content_template)

            # Add to list of tool specs for network
            tools.append(content_agent)

            # Add to list of tools for front man
            content_tools.append(tool_name)

        # Start out with the front man from the template, but replace him with what's made.
        front_man: Dict[str, Any] = tools[self.TEMPLATE_FRONT_MAN_INDEX]
        front_man = self.create_front_man(front_man, group, content_tools)
        # We don't need user prompts here
        del front_man["user_prompt"]
        tools[self.TEMPLATE_FRONT_MAN_INDEX] = front_man

        return agent_spec

    async def create_one_content_agent(self, file_name: str, tool_name: str, content_template: Dict[str, Any]) \
            -> Dict[str, Any]:
        """
        Creates a single agent node that sponsors one section of the content
        """
        # Asynchronously read the content of the file
        filepath = Path(self.files_directory) / file_name
        self.logger.info("Reading %s", filepath)
        async with aiofiles.open(filepath, "r") as my_file:
            file_content: str = await my_file.read()

        # Create the content agent spec by replacing strings in strategic places
        content_agent: Dict[str, Any] = deepcopy(content_template)
        string_replacements: Dict[str, Any] = {
            "one_content_file": tool_name,
            "content": file_content,
        }

        content_agent = self.filter_agent(content_agent, string_replacements)
        return content_agent

    def filter_agent(self, agent_spec: Dict[str, Any], replacements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Common filters
        """
        # Create a network spec so the ConfigFilters from neuro_san can work on it
        network_spec: Dict[str, Any] = {
            "tools": [agent_spec]
        }

        # Set up string replacements and include AAOSA stuff that we have to do
        # ourselves because we are creating a dictionary and not a hocon file.
        string_replacements: Dict[str, str] = {
            "aaosa_command": self.aaosa_defs.get("aaosa_command"),
            "aaosa_instructions": self.aaosa_defs.get("aaosa_instructions")
        }
        string_replacements.update(replacements)
        string_filter = StringCommonDefsConfigFilter(string_replacements)
        network_spec = string_filter.filter_config(network_spec)

        # Similarly set up dictionary value replacements
        dict_replacements: Dict[str, Any] = {
            "aaosa_call": self.aaosa_defs.get("aaosa_call"),
        }
        dict_filter = DictionaryCommonDefsConfigFilter(dict_replacements)
        network_spec = dict_filter.filter_config(network_spec)

        # Retrieve the modified agent spec
        return network_spec["tools"][0]

    def create_front_man(self, front_man: Dict[str, Any],
                         group: Dict[str, Any],
                         tools: List[str] = None) -> Dict[str, Any]:
        """
        Creates a front man
        """

        # Replace strings in the front man first
        string_replacements: Dict[str, Any] = {
            "one_group": group.get("name"),
            "group_description": group.get("description"),
            "structure_description": self.grouping_json.get("description"),
            "title": self.grouping_json.get("name"),
        }
        front_man = self.filter_agent(front_man, string_replacements)

        front_man["tools"] = tools

        return front_man

    async def reserve_leaf_networks(self, reservationist: Reservationist,
                                    name_to_network: Dict[str, Dict[str, Any]]) \
            -> Dict[Reservation, Dict[str, Any]]:
        """
        Creates reservations for each named network
        """
        deployments: Dict[Reservation, Dict[str, Any]] = {}

        for name, network in name_to_network.items():
            filtered_name: str = name.replace(" ", "_")
            reservation: Reservation = await reservationist.reserve(lifetime_in_seconds=self.LIFETIME,
                                                                    prefix=filtered_name)
            deployments[reservation] = network

        return deployments

    def make_group_network(self, reservations: List[Reservation]) -> Dict[str, Any]:
        """
        Creates a final front-man network for the rest.
        """

        agent_spec: Dict[str, Any] = deepcopy(self.network_template)
        tools: List[Dict[str, Any]] = agent_spec.get("tools")

        # We don't need the content node, we are using external networks for those.
        _ = tools.pop()

        # Make a list of the external networks for the reservations to reference as tools
        external_tools: List[str] = []
        for reservation in reservations:
            res_id: str = reservation.get_url()
            if not res_id.startswith("/") or not res_id.startswith("http"):
                res_id = "http://localhost/" + res_id
            external_tools.append(res_id)

        # Start out with the front man from the template, but replace him with what's made.
        front_man: Dict[str, Any] = tools[self.TEMPLATE_FRONT_MAN_INDEX]
        front_man = self.create_front_man(front_man, self.grouping_json, external_tools)
        front_man["function"]["description"] = front_man["user_prompt"]
        tools[self.TEMPLATE_FRONT_MAN_INDEX] = front_man

        return agent_spec

    def assemble_reservation_info(self, reservations: List[Reservation]) -> List[Dict[str, Any]]:
        """
        Assemble the list of networks available to the user.
        """
        reservation_info: List[Dict[str, Any]] = []
        for reservation in reservations:
            one_info: Dict[str, Any] = {
                "reservation_id": reservation.get_reservation_id(),
                "lifetime_in_seconds": reservation.get_lifetime_in_seconds(),
                "expiration_time_in_seconds": reservation.get_expiration_time_in_seconds(),
            }
            reservation_info.append(one_info)
        return reservation_info
