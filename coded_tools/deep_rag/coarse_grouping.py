
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
            if end_index > num_files:
                end_index = num_files
            file_groups.append(file_list[start_index:end_index])

        # Fill in the common args to be used across all file groups
        basis_args: Dict[str, Any] = {
            "files_directory": files_directory,
            "user_description": user_description,
            "grouping_constraints": grouping_constraints
        }

        _ = await self.do_subgroups_in_parallel(file_groups, basis_args, sly_data, tools_to_use)

        results: str = await self.process_group_results(sly_data)
        return results

    async def do_subgroups_in_parallel(self, file_groups: List[List[str]], basis_args: Dict[str, Any],
                                       sly_data: Dict[str, Any], tools_to_use: Dict[str, str]) -> str:

        # Create a single sly_data group_results entry so that parallel tasks have a place
        # to put their sly_data output without stomping on each other
        sly_data["group_results"] = []

        # Now create coroutines that will call rough_substructure on each group with data appropriate for the group
        coroutines: List[Future] = []
        for group_number, file_group in enumerate(file_groups):

            # Create a tool args dict specific to the iteration
            tool_args: Dict[str, Any] = deepcopy(basis_args)
            tool_args["file_list"] = file_group

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

        # Get tools we will call from role-keys
        rough_substructure: str = tools_to_use.get("rough_substructure", "rough_substructure")
        create_network: str = tools_to_use.get("create_network", "create_network")

        # Call rough_substructure
        one_grouping_json_str: str = await self.use_tool(tool_name=rough_substructure,
                                                         tool_args=tool_args,
                                                         sly_data=sly_data)
        one_grouping: Dict[str, Any] = JsonStructureParser().parse(one_grouping_json_str)

        # Call create_network
        create_network_args: Dict[str, Any] = {
            "files_directory": tool_args.get("files_directory"),
            "grouping_json": one_grouping,
            "group_number": group_number
        }
        result: str = await self.use_tool(tool_name=create_network, tool_args=create_network_args, sly_data=sly_data)

        return result

    def prepare_agent_reservations(self, sly_data: Dict[str, Any]):

        group_results: List[Dict[str, Any]] = sly_data.get("group_results")

        # Put the list of agent_reservations from each group into a single list
        sly_data["agent_reservations"] = []
        mid_level_networks: List[Dict[str, Any]] = []
        for group_result in group_results:

            reservation_info: List[Dict[str, Any]] = group_result.get("agent_reservations")

            # All the sub-agent networks will be the first items in the list, except for the last guy
            sly_data["agent_reservations"].extend(reservation_info[:-1])

            # The last one in the list will be the entry-point network, by convention
            mid_level_networks.append(reservation_info[-1])

        # Add the mid-level networks to the end of the list
        sly_data["agent_reservations"].extend(mid_level_networks)

    async def process_group_results(self, sly_data: Dict[str, Any]) -> str:

        self.prepare_agent_reservations(sly_data)

        group_results: List[Dict[str, Any]] = sly_data.get("group_results")

        # Early return situation if there is only one group.
        if len(group_results) == 1:
            # Use the aa_ prefix so that when keys come out in alphabetical order
            # the agent_reservations info will be the last thing spit out on command-line clients,
            # which will make the user's life easier.
            sly_data["aa_grouping_json"] = group_results[0].get("grouping_json")

        # Put the list of agent_reservations from each group into a single list
        reservation_info: List[Dict[str, Any]] = sly_data.get("agent_reservations")

        # By convention, the last entry in the reservation_info is the main entry point.
        entry: Dict[str, Any] = reservation_info[-1]
        entry_reservation_id: str = entry.get("reservation_id")
        entry_lifetime: str = entry.get("lifetime_in_seconds")

        output: str = f"The main agent to access your deep rag network is {entry_reservation_id}" + \
                      f"Hurry, it's only available for {entry_lifetime} seconds."
        return output
