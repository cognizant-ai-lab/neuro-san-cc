
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
from copy import deepcopy
from json import dumps
from logging import getLogger
from logging import Logger

from neuro_san.interfaces.coded_tool import CodedTool
from neuro_san.internals.graph.activations.branch_activation import BranchActivation
from neuro_san.internals.parsers.structure.json_structure_parser import JsonStructureParser

from coded_tools.deep_rag.create_networks import CreateNetworks


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
        empty: Dict[str, Any] = {}

        # Load stuff from args into local variables
        tools_to_use: Dict[str, str] = args.get("tools", empty)

        file_groups: List[List[str]] = self.create_file_groups(args)

        # Fill in the common args to be used across all file groups
        basis_args: Dict[str, Any] = {
            "files_directory": args.get("files_directory", ""),
            "user_description": args.get("user_description", ""),
            "grouping_constraints": args.get("grouping_constraints", "")
        }

        _ = await self.do_subgroups_in_parallel(file_groups, basis_args, sly_data, tools_to_use)

        results: str = await self.process_group_results(sly_data)
        return results

    def create_file_groups(self, args: Dict[str, Any]) -> List[List[str]]:
        """
        Break the file list into manageable groups
        :param args: A dictionary of arguments from the invocation of this CodedTool
        :return: A list of lists of file names
        """

        empty_list: List[str] = []

        file_list: List[str] = args.get("file_list", empty_list)
        num_files: int = len(file_list)
        max_group_size: int = int(args.get("max_group_size", 42))

        # Assume at first that this will all fit in a single group
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
            end_index = min(end_index, num_files)
            file_groups.append(file_list[start_index:end_index])

        return file_groups

    async def do_subgroups_in_parallel(self, file_groups: List[List[str]], basis_args: Dict[str, Any],
                                       sly_data: Dict[str, Any], tools_to_use: Dict[str, str]) -> str:

        """
        Call rough_substructure and create_networks on each group in parallel
        The results of the individually created group networks will be in sly_data's "group_results" key.
        :param file_groups: A list of lists of file names
        :param basis_args: A dictionary of arguments common to all file groups
        :param sly_data: A dictionary whose keys are defined by the agent hierarchy,
                but whose values are meant to be kept out of the chat stream.
                This dictionary is largely to be treated as read-only.
                It is possible to add key/value pairs to this dict that do not
                yet exist as a bulletin board, as long as the responsibility
                for which coded_tool publishes new entries is well understood
                by the agent chain implementation and the coded_tool implementation
                adding the data is not invoke()-ed more than once.
        :param tools_to_use: A dictionary of tools to use
        :return: A list of string results from all the parallel tasks.
        """
        logger: Logger = getLogger(self.__class__.__name__)

        # Create a single sly_data group_results entry so that parallel tasks have a place
        # to put their sly_data output without stomping on each other
        sly_data["group_results"] = []

        # Now create coroutines that will call rough_substructure on each group with data appropriate for the group
        coroutines: List[Future] = []
        logger.info("Processing %d file groups", len(file_groups))
        for group_number, file_group in enumerate(file_groups):

            # Create a tool args dict specific to the iteration
            tool_args: Dict[str, Any] = deepcopy(basis_args)
            tool_args["file_list"] = file_group

            logger.info("Processing group %d with list: %s", group_number,
                        dumps(file_group, indent=4, sort_keys=True))

            # Add an empty entry for each group to the group_results
            sly_data["group_results"].append({})

            # Add a coroutine for the file group to the list
            coroutines.append(self.do_one_subgroup_in_parallel(group_number, tool_args, sly_data, tools_to_use))

        # Call the rough_substructure and create_networks tools on each group in parallel
        # The results of the mid- to leaf-level group networks will be in sly_data's group_results.
        results: List[str] = await gather(*coroutines)

        return results

    async def do_one_subgroup_in_parallel(self, group_number: int,
                                          tool_args: Dict[str, Any],
                                          sly_data: Dict[str, Any],
                                          tools_to_use: Dict[str, str]) -> str:
        """
        Call rough_substructure and create_networks in parallel on a single file grouping.
        :param group_number: The index of the file group being processed
        :param tool_args: The basis arguments to be passed to rough_substructure and create_networks
        :param sly_data: The sly_data dictionary for the instantiation of the coded tool
        :param tools_to_use: The dictionary of tools to be called
        """

        # Get tools we will call from role-keys
        rough_substructure: str = tools_to_use.get("rough_substructure", "rough_substructure")
        create_network: str = tools_to_use.get("create_network", "create_network")

        # Call rough_substructure
        one_grouping_json_str: str = await self.use_tool(tool_name=rough_substructure,
                                                         tool_args=tool_args,
                                                         sly_data=sly_data)
        one_grouping: Dict[str, Any] = JsonStructureParser().parse_structure(one_grouping_json_str)

        # Call create_network
        create_network_args: Dict[str, Any] = {
            "files_directory": tool_args.get("files_directory"),
            "grouping_json": one_grouping,
            "group_number": group_number
        }
        result: str = await self.use_tool(tool_name=create_network, tool_args=create_network_args, sly_data=sly_data)

        return result

    def prepare_agent_reservations(self, sly_data: Dict[str, Any]) -> None:
        """
        Put the list of agent_reservations from the parallel calls to create_network
        into a single list.
        :param sly_data: The sly_data dictionary for the instantiation of the coded tool
                        where we will put our results.  We expect "group_results" to have
                        already been filled in.
        """

        group_results: List[Dict[str, Any]] = sly_data.get("group_results")

        # Put the list of agent_reservations from each group into a single list
        sly_data["agent_reservations"] = []
        mid_level_networks: List[Dict[str, Any]] = []
        for group_number, group_result in enumerate(group_results):

            reservation_info: List[Dict[str, Any]] = group_result.get("agent_reservations")

            logger: Logger = getLogger(self.__class__.__name__)
            if not reservation_info:
                logger.warning("No agent_reservations found for group %d", group_number)
                continue

            if not isinstance(reservation_info, list):
                logger.warning("agent_reservations found for group %d is not a list", group_number)
                continue

            # All the sub-agent networks will be the first items in the list, except for the last guy
            sly_data["agent_reservations"].extend(reservation_info[:-1])

            # The last one in the list will be the entry-point network, by convention
            mid_level_networks.append(reservation_info[-1])

        # Add the mid-level networks to the end of the list
        sly_data["agent_reservations"].extend(mid_level_networks)

    async def process_group_results(self, sly_data: Dict[str, Any]) -> str:
        """
        Integrate the results from all the calls to the rough_substructure and create_network tools
        into a single whole.

        :param sly_data: The sly_data dictionary for the instantiation of the coded tool
                        where we will put our results.  We expect "group_results" to have
                        already been filled in.
        :return: String output to return as tool output
        """

        self.prepare_agent_reservations(sly_data)

        group_results: List[Dict[str, Any]] = sly_data.get("group_results")

        # Early return situation if there is only one group.
        if len(group_results) == 1:
            # Use the aa_ prefix so that when keys come out in alphabetical order
            # the agent_reservations info will be the last thing spit out on command-line clients,
            # which will make the user's life easier.
            sly_data["aa_grouping_json"] = group_results[0].get("grouping_json")
        else:
            # Use the aa_ prefix so that when keys come out in alphabetical order
            # the agent_reservations info will be the last thing spit out on command-line clients,
            # which will make the user's life easier.
            grouping_json_list: List[Dict[str, Any]] = []
            for group_result in group_results:
                grouping_json_list.append(group_result.get("grouping_json"))
            sly_data["aa_grouping_json"] = grouping_json_list

        # Put the list of agent_reservations from each group into a single list
        reservation_info: List[Dict[str, Any]] = sly_data.get("agent_reservations")

        output: str = CreateNetworks.create_output(reservation_info)
        return output
