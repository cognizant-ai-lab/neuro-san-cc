
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
from asyncio import Lock
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
    into smaller groups where each subgroup of files can be digested by a single pass to the
    rough_substructure agent to create one specialist leaf network.

    These specialist leaf networks are then assembled into groups of 6 for a higher-level network.
    This agglomeration recurses upwards until a single entry-point network is created.

    Note: We also derive from BranchActivation so that we can employ other tools within
    the agent hierarchy.
    """

    # These constants could conceivably be made into args specified by the agent network.
    MAX_GROUP_SIZE: int = 6
    MAX_FILES_PER_GROUP: int = 7
    MAX_RETRIES: int = 3

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
        empty_list: List[str] = []

        # Load stuff from args into local variables
        tools_to_use: Dict[str, str] = args.get("tools", empty)

        file_list: List[str] = args.get("file_list", empty_list)
        max_group_size: int = int(args.get("max_group_size", 42))
        max_group_size = min(max_group_size, self.MAX_GROUP_SIZE * self.MAX_FILES_PER_GROUP)
        file_groups: List[List[str]] = self.create_groups(file_list, max_group_size)

        # Fill in the common args to be used across all file groups
        basis_args: Dict[str, Any] = {
            "files_directory": args.get("files_directory", ""),
            "user_description": args.get("user_description", ""),
            "grouping_constraints": args.get("grouping_constraints", "")
        }

        # Create a single sly_data group_results entry so that parallel tasks have a place
        # to put their sly_data output without stomping on each other
        sly_data["group_results"] = []
        sly_data["num_groups"] = 0
        sly_data["lock"] = Lock()
        sly_data["agent_reservations"] = []

        _ = await self.do_subgroups_in_parallel(file_groups, basis_args, sly_data, tools_to_use)

        results: str = await self.process_group_results(sly_data, tools_to_use)
        return results

    def create_groups(self, item_list: List[Any], max_group_size: int) -> List[List[Any]]:
        """
        Break a large list into manageable groups
        :param args: A dictionary of arguments from the invocation of this CodedTool
        :param max_group_size: The maximum number of items allowed in a single group
        :return: A list of lists of items names
        """

        num_items: int = len(item_list)

        # Assume at first that this will all fit in a single group
        num_groups: int = 1
        items_per_group: int = num_items
        if num_items > max_group_size:
            # This won't fit into a single group. Break it up as evenly as possible
            num_groups = int(num_items / max_group_size)
            if num_items % max_group_size != 0:
                num_groups += 1
            items_per_group = int(num_items / num_groups)

        # Break the item list into manageable groups
        item_groups: List[List[Any]] = []
        for group_index in range(num_groups):
            start_index: int = group_index * items_per_group
            end_index: int = start_index + items_per_group
            end_index = min(end_index, num_items)
            item_groups.append(item_list[start_index:end_index])

        return item_groups

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

        # Now create coroutines that will call rough_substructure on each group with data appropriate for the group
        coroutines: List[Future] = []
        logger.info("Processing %d file groups", len(file_groups))
        for file_group in file_groups:

            # Create a tool args dict specific to the iteration
            tool_args: Dict[str, Any] = deepcopy(basis_args)
            tool_args["file_list"] = file_group

            group_number: int = await self.new_group(sly_data)

            sly_data["num_groups"] = len(sly_data["group_results"])

            # Add a coroutine for the file group to the list
            coroutines.append(self.do_one_subgroup_in_parallel(group_number, tool_args, sly_data, tools_to_use))

        # Call the rough_substructure and create_networks tools on each group in parallel
        # The results of the mid- to leaf-level group networks will be in sly_data's group_results.
        results: List[str] = await gather(*coroutines)

        return results

    # pylint: disable=too-many-locals
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
        logger: Logger = getLogger(self.__class__.__name__)

        # Get tools we will call from role-keys
        rough_substructure: str = tools_to_use.get("rough_substructure", "rough_substructure")
        create_network: str = tools_to_use.get("create_network", "create_network")

        file_list: List[str] = tool_args.get("file_list")

        logger.info("Processing group %d with list: %s", group_number, dumps(file_list, indent=4, sort_keys=True))

        # Call rough_substructure
        done: bool = False
        retries_left: int = self.MAX_RETRIES
        while not done:
            one_grouping_json_str: str = await self.use_tool(tool_name=rough_substructure,
                                                             tool_args=tool_args,
                                                             sly_data=sly_data)
            one_grouping: Dict[str, Any] = JsonStructureParser().parse_structure(one_grouping_json_str)
            groups: List[Dict[str, Any]] = None
            if one_grouping is not None:
                groups = one_grouping.get("groups")

            done = self.verify_grouping_constraints(groups, file_list)

            if not done:
                retries_left -= 1
                if retries_left <= 0:
                    logger.info("Constraints not met after %d retries.", self.MAX_RETRIES)
                    done = True

        if retries_left <= 0:
            return await self.split_up_list(group_number, tool_args, sly_data, tools_to_use)

        # Call create_network
        create_network_args: Dict[str, Any] = {
            "files_directory": tool_args.get("files_directory"),
            "grouping_json": one_grouping,
            "group_number": group_number
        }
        result: str = await self.use_tool(tool_name=create_network, tool_args=create_network_args, sly_data=sly_data)

        return result

    async def split_up_list(self, group_number: int, tool_args: Dict[str, Any],
                            sly_data: Dict[str, Any], tools_to_use: Dict[str, str]) -> str:
        """
        Split a file list in two
        :param group_number: The index of the file group being processed
        :param tool_args: The basis arguments to be passed to rough_substructure and create_networks
        :param sly_data: The sly_data dictionary for the instantiation of the coded tool
        :param tools_to_use: The dictionary of tools to be called
        """
        logger: Logger = getLogger(self.__class__.__name__)

        file_list: List[str] = tool_args.get("file_list")

        # Break up the list
        num_files: int = int(len(file_list) / 2)
        group_one: List[str] = file_list[0:num_files]   # end index is not included
        group_two: List[str] = file_list[num_files:-1]
        logger.info("Splitting list into two groups of %d and %d", len(group_one), len(group_two))

        new_group_number: int = await self.new_group(sly_data)

        # Do the two groups in parallel
        coroutines: List[Future] = []

        tool_args_one: Dict[str, Any] = deepcopy(tool_args)
        tool_args_one["file_list"] = group_one
        coroutines.append(self.do_one_subgroup_in_parallel(group_number, tool_args_one, sly_data, tools_to_use))

        tool_args_two: Dict[str, Any] = deepcopy(tool_args)
        tool_args_two["file_list"] = group_two
        coroutines.append(self.do_one_subgroup_in_parallel(new_group_number, tool_args_two, sly_data, tools_to_use))

        result: List[str] = await gather(*coroutines)
        return str(result)

    def verify_grouping_constraints(self, groups: List[Dict[str, Any]], file_list: List[str]) -> bool:
        """
        Verify that the grouping constraints are met.
        :param groups: A list of groups
        :param file_list: A list of files
        :return: True if the constraints are met, False otherwise
        """

        logger: Logger = getLogger(self.__class__.__name__)

        if groups is None:
            logger.info("Constraints not met. groups is None.")
            return False

        # Verify the grouping constraints.
        if len(groups) > self.MAX_GROUP_SIZE:
            logger.info("Constraints not met. Too many groups (%d).", len(groups))
            return False

        # Verify the file-per-group constraints.
        for group in groups:
            files: Dict[str, Any] = group.get("files")
            if len(files) > self.MAX_FILES_PER_GROUP:
                logger.info("Too many files in group (%d). Constraints not met.", len(files))
                return False

        # Verify that every file is in one group
        for file in file_list:
            found: bool = False
            for group in groups:
                files: Dict[str, Any] = group.get("files")
                if file in files:
                    found = True
                    break
            if not found:
                logger.info("Constraints not met. File %s not found in any group", file)
                return False

        return True

    def prepare_agent_reservations(self, sly_data: Dict[str, Any],
                                   new_group_numbers: List[int] = None) -> List[Dict[str, Any]]:
        """
        Put the list of agent_reservations from the parallel calls to create_network
        into a single list.
        :param sly_data: The sly_data dictionary for the instantiation of the coded tool
                        where we will put our results.  We expect "group_results" to have
                        already been filled in.
        :param new_group_numbers: A list of new group numbers to process.  If None then will process all.
        :return: A list of the new mid-level networks that will need to be grouped together
        """

        group_results: List[Dict[str, Any]] = sly_data.get("group_results")

        use_group_numbers: List[int] = new_group_numbers

        # Put the list of agent_reservations from each group into a single list
        mid_level_networks: List[Dict[str, Any]] = []
        for group_number in use_group_numbers:

            group_result: Dict[str, Any] = group_results[group_number]
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

        return mid_level_networks

    async def process_group_results(self, sly_data: Dict[str, Any], tools_to_use: Dict[str, Any],
                                    new_group_numbers: List[int] = None) -> str:
        """
        Integrate the results from all the calls to the rough_substructure and create_network tools
        into a single whole.

        :param sly_data: The sly_data dictionary for the instantiation of the coded tool
                        where we will put our results.  We expect "group_results" to have
                        already been filled in.
        :param tools_to_use: A dictionary of tools to use
        :param new_group_numbers: A list of new group numbers to process.  If None then will process all.
        :return: String output to return as tool output
        """
        group_results: List[Dict[str, Any]] = sly_data.get("group_results")

        use_group_numbers: List[int] = new_group_numbers
        if use_group_numbers is None:
            use_group_numbers = list(range(len(group_results)))

        # Contains only new mid-level networks
        mid_level_networks: List[Dict[str, Any]] = self.prepare_agent_reservations(sly_data, use_group_numbers)

        # Use the aa_ prefix so that when keys come out in alphabetical order
        # the agent_reservations info will be the last thing spit out on command-line clients,
        # which will make the user's life easier in terms of finding the main network to call.
        grouping_json_list: List[Dict[str, Any]] = []
        for group_result in group_results:
            grouping_json_list.append(group_result.get("grouping_json"))
        sly_data["aa_grouping_json"] = grouping_json_list

        if len(mid_level_networks) > 1:

            # Create a master list of mid-level group information
            mid_level_groups: List[Dict[str, Any]] = []
            mid_level_network: Dict[str, Any] = None
            for index, mid_level_network in enumerate(mid_level_networks):

                group_number: int = use_group_numbers[index]
                group_result: Dict[str, Any] = group_results[group_number]

                mid_level_group: Dict[str, Any] = {
                    "reservation_dict": mid_level_network,
                    "grouping_json": group_result.get("grouping_json")
                }
                mid_level_groups.append(mid_level_group)

            # Create groupings of groups
            mid_level_groupings: List[List[Dict[str, Any]]] = self.create_groups(mid_level_groups, self.MAX_GROUP_SIZE)
            await self.create_groups_of_groups(mid_level_groupings, sly_data, tools_to_use)

        # Put the list of agent_reservations from each group into a single list
        reservation_info: List[Dict[str, Any]] = sly_data.get("agent_reservations")

        output: str = CreateNetworks.create_output(reservation_info)
        return output

    async def create_groups_of_groups(self, mid_level_groupings: List[List[Dict[str, Any]]],
                                      sly_data: Dict[str, Any], tools_to_use: Dict[str, Any]) -> str:
        """
        Create groupings of groups
        :param mid_level_groupings: The list of groupings of groups
        :param sly_data: The sly_data dictionary for the instantiation of the coded tool
        :param tools_to_use: A dictionary of tools to use
        :return: What to say to the user
        """
        create_network: str = tools_to_use.get("create_network", "create_network")

        # Assembles a list of coroutines to do in parallel
        coroutines: List[Future] = []
        new_group_numbers: List[int] = []

        mid_level_grouping: List[Dict[str, Any]] = None
        for mid_level_grouping in mid_level_groupings:

            # Accumulate groups for a higher-level network descriptions
            high_level_groups: List[Dict[str, Any]] = []

            mid_level_group: Dict[str, Any] = None
            for mid_level_group in mid_level_grouping:

                reservation_dict: Dict[str, Any] = mid_level_group.get("reservation_dict")
                grouping_json: Dict[str, Any] = mid_level_group.get("grouping_json")

                new_group: Dict[str, Any] = {
                    "description": grouping_json.get("description"),
                    "name": grouping_json.get("name"),
                    "reservation": reservation_dict
                }
                high_level_groups.append(new_group)

            high_level_grouping: Dict[str, Any] = {
                "name": "group_of_groups",              # DEF - Can get an agent to do better
                "description": "Grouping of groups",    # DEF - Can get an agent to do better
                "groups": high_level_groups
            }

            new_group_number: int = await self.new_group(sly_data)
            new_group_numbers.append(new_group_number)

            # Prepare call to create_network
            create_network_args: Dict[str, Any] = {
                "files_directory": "",
                "grouping_json": high_level_grouping,
                "group_number": new_group_number
            }

            coroutines.append(self.use_tool(tool_name=create_network, tool_args=create_network_args, sly_data=sly_data))

        # Run all the coroutines in parallel
        await gather(*coroutines)

        # Recurse to process the results.
        results: str = await self.process_group_results(sly_data, tools_to_use, new_group_numbers)
        return results

    async def new_group(self, sly_data: Dict[str, Any]) -> int:
        """
        Create a new group
        :param sly_data: The sly_data dictionary for the instantiation of the coded tool
        :return: The index of the new group
        """

        new_group_number: int = 0
        # Potential, but unlikely race condition
        lock: Lock = sly_data.get("lock")
        async with lock:

            group_results: List[Dict[str, Any]] = sly_data.get("group_results")
            if group_results is None:
                # Initialize
                sly_data["group_results"] = []
                sly_data["num_groups"] = 0

            new_group_number: int = len(group_results)

            # Add a slot to the group_results list for future asynchronous population.
            sly_data["group_results"].append({})
            sly_data["num_groups"] = new_group_number + 1

        logger: Logger = getLogger(self.__class__.__name__)
        logger.info("Created new group %d", new_group_number)

        return new_group_number
