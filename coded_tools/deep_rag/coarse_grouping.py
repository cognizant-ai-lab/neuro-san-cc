
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

from asyncio import Future
from asyncio import gather
from logging import getLogger
from logging import Logger

from neuro_san.interfaces.coded_tool import CodedTool
from neuro_san.internals.graph.activations.branch_activation import BranchActivation
from neuro_san.internals.parsers.structure.json_structure_parseri import JsonStructureParser


class CoarseGrouping(BranchActivation, CodedTool):
    """
    CodedTool implementation that potentially breaks a large list of file references
    into smaller groups where each subgroup can be digested by a single pass to the
    rough_substructure agent.
    """

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
        logger: Logger = getLogger(self.__class__.__name__)
        _ = logger

        empty: Dict[str, Any] = {}
        empty_list: List[str] = []

        # Load stuff from args into local variables
        file_list: List[str] = args.get("file_list", empty_list)
        num_files: int = len(file_list)
        files_directory: str = args.get("files_directory", "")
        user_description: str = args.get("user_description", "")
        grouping_constraints: str = args.get("grouping_constraints")
        max_group_size: int = int(args.get("max_group_size", 42))
        tools_to_use: Dict[str, str] = args.get("tools", empty)

        # Assume this will all fit in a single group
        num_groups: int = 1
        files_per_group: int = num_files
        if num_files > max_group_size:
            # This won't fit into a single group. Break it up as evenly as possible
            num_groups = int(num_files / max_group_size)
            if num_files % max_group_size != 0:
                num_groups += 1
            files_per_group = int(num_files / num_groups)

        # Break the file list into manageable groups
        file_groups: List[List[str]] = []
        for group_index in range(num_groups):
            start_index: int = group_index * files_per_group
            end_index: int = start_index + files_per_group
            if end_index > num_files:
                end_index = num_files
            file_groups.append(file_list[start_index:end_index])

        # Now create coroutines that will call rough_substructure on each group with data appropriate for the group
        coroutines: List[Future] = []
        for group_number, file_group in enumerate(file_groups):
            tool_args: Dict[str, Any] = {
                "file_list": file_group,
                "files_directory": files_directory,
                "user_description": user_description,
                "grouping_constraints": grouping_constraints
            }
            coroutines.append(self.do_one_subgroup_in_parallel(group_number, tool_args, sly_data, tools_to_use))

        # Call the rough_substructure tool on each group in parallel
        results: List[str] = await gather(*coroutines)

        return str(results)

    async def do_one_subgroup_in_parallel(self, group_number: int,
                                          tool_args: Dict[str, Any],
                                          sly_data: Dict[str, Any],
                                          tools_to_use: Dict[str, str]) -> str:

        # Get tools we will call from role-keys
        rough_substructure: str = tools_to_use.get("rough_substructure", "rough_substructure")
        create_network: str = tools_to_use.get("create_network", "create_network")

        one_grouping_json_str: str = await self.use_tool(tool_name=rough_substructure,
                                                         tool_args=tool_args,
                                                         sly_data=sly_data)
        one_grouping: Dict[str, Any] = JsonStructureParser().parse(one_grouping_json_str)

        create_network_args: Dict[str, Any] = {
            "files_directory": tool_args.get("files_directory"),
            "grouping_json": one_grouping
        }
        result: str = await self.use_tool(tool_name=create_network, tool_args=create_network_args, sly_data=sly_data)

        return result
