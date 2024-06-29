"""Memory system classes."""

from __future__ import annotations  # remove this from python 3.11

import random
from typing import Literal

from .utils import merge_lists


class Memory:
    """Memory class.

    At the moment, the memory system is a simple Python list of memories. In the future,
    a more suitable python object will be used to represent the graph structure of the
    memories.

    Attributes:
        type: episodic, semantic, short, or working
        entries: list of memories
        capacity: memory capacity
        _frozen: whether the memory system is frozen or not


    """

    def __init__(self, capacity: int, memories: list[list] = []) -> None:
        """

        Args:
            capacity: memory capacity
            memories: memories that can already be added from the beginning, if None,
                then it's an empty memory system.

        """
        self.entries = []
        self.capacity = capacity
        assert self.capacity >= 0
        self._frozen = False

        if memories:
            for mem in memories:
                check, error_msg = self.can_be_added(mem)
                if not check:
                    raise ValueError(error_msg)
                else:
                    self.add(mem)

    def __iter__(self):
        return iter(self.entries[:])

    def __len__(self):
        return len(self.entries)

    def __add__(self, other):
        entries = self.entries + other.entries
        return Memory(self.capacity + other.capacity, entries)

    def can_be_added(self, mem) -> tuple[bool, str | None]:
        """Check if a memory can be added to the system or not.

        Returns:
            True or False
            error_msg

        """
        if self.capacity == 0:
            return False, "The memory system capacity is 0!"

        if self._frozen:
            return False, "The memory system is frozen!"

        return True, ""

    def add(self, mem: list) -> None:
        """Add memory to the memory system.

        There is no sorting done. It's just appended to the end.

        Args:
           mem: A memory as a quadraple: [head, relation, tail, num]

        """
        check, error_msg = self.can_be_added(mem)
        if not check:
            raise ValueError(error_msg)
        self.entries.append(mem)

        if self.size > self.capacity:
            raise ValueError(f"Something went wrong. {self.size} > {self.capacity}.")

    def can_be_forgotten(self, mem: list) -> tuple[bool, str]:
        """Check if a memory can be added to the system or not.

        Args:
            mem: A memory as a quadraple: [head, relation, tail, num]

        Returns:
            True or False
            error_msg

        """
        if self.capacity == 0:
            return False, "The memory system capacity is 0!"

        if self.size == 0:
            return False, "The memory system is empty!"

        if self._frozen:
            return False, "The memory system is frozen!"

        if mem not in self.entries:
            return False, f"{mem} is not in the memory system!"

        return True, None

    def forget(self, mem: list) -> None:
        """forget the given memory.

        Args:
            mem: A memory as a quadraple: [head, relation, tail, num], where `num` is
                either a list of an int.

        """
        self.entries.remove(mem)

    def forget_all(self) -> None:
        """Forget everything in the memory system!"""
        if self.capacity == 0:
            error_msg = "The memory system capacity is 0. Can't forget all."
            raise ValueError(error_msg)

        if self.is_frozen:
            error_msg = "The memory system is frozen. Can't forget all. Unfreeze first."
            raise ValueError(error_msg)

        else:
            self.entries = []

    def has_memory(self, mem: list) -> bool:
        """Check if a memory is in the memory system.

        Args:
            mem: A memory as a quadraple: [head, relation, tail, object]

        Returns:
            True or False

        """
        return mem in self.entries

    @property
    def is_empty(self) -> bool:
        """Return true if empty."""
        return len(self.entries) == 0

    @property
    def is_full(self) -> bool:
        """Return true if full."""
        return len(self.entries) == self.capacity

    @property
    def is_frozen(self) -> bool:
        """Is frozen?"""
        return self._frozen

    @property
    def size(self) -> int:
        """Get the size (number of filled entries) of the memory system."""
        return len(self.entries)

    def freeze(self) -> None:
        """Freeze the memory so that nothing can be added / deleted."""
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze the memory so that something can be added / deleted."""
        self._frozen = False

    def forget_random(self) -> None:
        """Forget a memory in the memory system in a uniform-randomly."""
        mem = random.choice(self.entries)
        self.forget(mem)

    def increase_capacity(self, increase: int) -> None:
        """Increase the capacity.

        Args:
            increase: the amount of entries to increase.

        """
        assert isinstance(increase, int) and (not self.is_frozen)
        self.capacity += increase

    def decrease_capacity(self, decrease: int) -> None:
        """decrease the capacity.

        Args:
            decrease: the amount of entries to decrease.

        """
        assert (
            isinstance(decrease, int)
            and (self.capacity - decrease >= 0)
            and (not self.is_frozen)
        )
        self.capacity -= decrease

    def to_list(self) -> list[list]:
        """Return the memories as a list of lists.

        Returns:
            a list of lists

        """
        return self.entries

    def query(self, query: list) -> Memory:
        """Query memory.

        Args:
            query: a quadruple, where each element can be "?". e.g.,
                ["bob", "atlocation", "?", "?], ["?", "atlocation", "officeroom", "?"]
                "?" is used to match any value.

        Returns:


        """
        assert len(query) == 4
        mems_found = []

        for mem in self.to_list():
            if (query[0] == "?") or (query[0] == mem[0]):
                if (query[1] == "?") or (query[1] == mem[1]):
                    if (query[2] == "?") or (query[2] == mem[2]):
                        if (query[3] == "?") or (set(query[3]).issubset(set(mem[3]))):
                            mems_found.append(mem)

        return Memory(len(mems_found), mems_found)

    def retrieve_random_memory(self) -> list:
        """Retrieve a random memory from the memory system.

        Returns:
            random_memory: A random memory from the memory system

        """
        return random.choice(self.to_list())

    def retrieve_memory_by_qualifier(
        self,
        qualifier: str,
        qualifier_object_type: Literal["list", "int"],
        select_by: Literal["max", "min"],
        list_select_by: Literal["max", "min"] | None = None,
    ) -> list | None:
        """Retrieve a memory based on a qualifier value.

        Args:
            qualifier: The qualifier to search for
            qualifier_object_type: The type of the qualifier object
            select_by: The selection method to use when comparing qualifier values
            list_select_by: The selection method to use when comparing list qualifier
                values

        Returns:
            desired_memory: The memory with the desired qualifier value

        """

        def get_qualifier_value(memory):
            for element in memory:
                if isinstance(element, dict) and qualifier in element:
                    return element[qualifier]
            return None

        # Initialize variables to track the memory with the desired qualifier value
        if select_by == "min":
            desired_value = float("inf")
        elif select_by == "max":
            desired_value = float("-inf")
        desired_memory = None

        # Comparison function to update the desired memory and value
        def compare_and_update(value, memory):
            nonlocal desired_value, desired_memory
            if (select_by == "min" and value < desired_value) or (
                select_by == "max" and value > desired_value
            ):
                desired_value = value
                desired_memory = memory

        # Iterate over each memory and update the desired memory based on the qualifier type
        for memory in self.to_list():
            qualifier_value = get_qualifier_value(memory)
            if qualifier_value is not None:
                if qualifier_object_type == "list":
                    if list_select_by == "max":
                        value = max(qualifier_value)
                    elif list_select_by == "min":
                        value = min(qualifier_value)
                else:  # qualifier_object_type == "int"
                    value = qualifier_value
                compare_and_update(value, memory)

        return desired_memory


class ShortMemory(Memory):
    """Short-term memory class."""

    def __init__(self, capacity: int, memories: list[list] | None = None) -> None:
        super().__init__(capacity, memories)

    def can_be_added(self, mem: list) -> tuple[bool, str | None]:
        """Check if a memory can be added to the short-term memory system.

        Args:
            mem: A memory as a quadraple: [head, relation, tail, current_time]

        Returns:
            True or False, error_msg

        """
        check, error_msg = super().can_be_added(mem)
        if not check:
            return check, error_msg

        if "current_time" not in mem[-1]:
            return False, "The memory should have current_time!"

        if self.is_full:
            for entry in self.entries:
                if entry[:-1] == mem[:-1]:
                    return True, None

            return False, "The memory system is full!"

        return True, None

    @staticmethod
    def ob2short(ob: list) -> list:
        """Turn an observation into an short memory.

        This is done by adding the qualifier "current_time" to the observation.

        Args:
            ob: An observation as a quadruple: [head, relation, tail, current_time]

        Returns:
            mem: A short-term memory as a quadruple: [head, relation, tail,
                {"current_time": int}]

        """
        mem = ob[:-1] + [{"current_time": ob[-1]}]

        return mem

    @staticmethod
    def short2epi(short: list) -> list:
        """Turn a short memory into a episodic memory.

        This is done by simply copying the short memory, and changing the qualifier
        "current_time" to "timestamp".

        Args:
            short: A short memory as a quadruple: [head, relation, tail,
                {"current_time": int}]

        Returns:
            epi: An episodic memory as a quadruple: [head, relation, tail, {"timestamp":
            [int]}]

        """
        epi = short[:-1] + [{"timestamp": [short[-1]["current_time"]]}]

        return epi

    @staticmethod
    def short2sem(short: list) -> list:
        """Turn a short memory into a semantic memory.

        Args:
            short: A short memory as a quadruple: [head, relation, tail,
                {"current_time": int}]

        Returns:
            sem: A semantic memory as a quadruple: [head, relation, tail,
                {"strength": int}]

        """
        sem = short[:-1] + [{"strength": 1}]

        return sem


class LongMemory(Memory):
    """Long-term memory class."""

    def __init__(self, capacity: int, memories: list[list] | None = None) -> None:
        super().__init__(capacity, memories)

    def can_be_added(self, mem: list) -> tuple[bool, str | None]:
        """Check if a memory can be added to the long-term memory system.

        Args:
            mem: A memory as a quadraple: [head, relation, tail, qualifiers]
            type: "episodic" or "semantic"

        Returns:
            True or False, error_msg

        """
        check, error_msg = super().can_be_added(mem)
        if not check:
            return check, error_msg

        if not set(mem[-1]).issubset(set(["timestamp", "strength"])):
            return False, "The memory should have timestamp or strength!"

        if self.is_full:
            for entry in self.entries:
                if entry[:-1] == mem[:-1]:
                    return True, None

            return False, "The memory system is full!"

        else:
            return True, None

    def add(self, mem: list) -> None:
        """Append a memory to the long-term memory system.

        Args:
            mem: A memory as a quadraple: [head, relation, tail, qualifiers]
        """
        assert self.can_be_added(mem)[0]

        added = False

        for entry in self.entries:
            if entry[:-1] == mem[:-1]:
                # Merge 'timestamp' values if present in both dictionaries
                if "timestamp" in entry[-1] and "timestamp" in mem[-1]:
                    entry[-1]["timestamp"] = sorted(
                        entry[-1]["timestamp"] + mem[-1]["timestamp"]
                    )
                elif "timestamp" in entry[-1]:
                    pass
                elif "timestamp" in mem[-1]:
                    entry[-1]["timestamp"] = mem[-1]["timestamp"]

                # Sum 'strength' values if present in both dictionaries
                if "strength" in entry[-1] and "strength" in mem[-1]:
                    entry[-1]["strength"] = entry[-1]["strength"] + mem[-1]["strength"]
                elif "strength" in entry[-1]:
                    pass
                elif "strength" in mem[-1]:
                    entry[-1]["strength"] = mem[-1]["strength"]

                added = True
                break

        if not added:
            super().add(mem)

    def forget_by_selection(
        self, selection: Literal["oldest", "latest", "weakest", "strongest"]
    ) -> None:
        """Forget a memory by selection.

        Args:
            selection: The selection method to use when forgetting a memory
        """

        if selection == "oldest":
            mem_oldest = self.retrieve_memory_by_qualifier(
                "timestamp", "list", "min", "max"
            )

            if mem_oldest is None:
                raise ValueError("There is no 'timestamp' key in any memory.")
            self.forget(mem_oldest)

        elif selection == "latest":
            mem_latest = self.retrieve_memory_by_qualifier(
                "timestamp", "list", "max", "max"
            )

            if mem_latest is None:
                raise ValueError("There is no 'timestamp' key in any memory.")
            self.forget(mem_latest)

        elif selection == "weakest":
            mem_weakest = self.retrieve_memory_by_qualifier("strength", "int", "min")

            if mem_weakest is None:
                raise ValueError("There is no 'strength' key in any memory.")
            self.forget(mem_weakest)

        elif selection == "strongest":
            mem_strongest = self.retrieve_memory_by_qualifier("strength", "int", "max")

            if mem_strongest is None:
                raise ValueError("There is no 'strength' key in any memory.")
            self.forget(mem_strongest)

        else:
            raise ValueError(
                "Invalid selection. Please choose from "
                "'oldest', 'latest', 'weakest', 'strongest'."
            )

    def count_memories(self) -> tuple[int, int]:
        """Count the memories with qualifiers, "timestamp" and "strength", respectively.

        Returns:
            number of "timestamp" memories, number of "current_time" memories

        """
        num_timestamps = 0
        num_strengths = 0

        for mem in self.to_list():
            if "timestamp" in mem[-1]:
                num_timestamps += 1
            if "strength" in mem[-1]:
                num_strengths += 1

        return num_timestamps, num_strengths

    def pretrain_semantic(
        self,
        semantic_knowledge: list[list],
    ) -> None:
        """Pretrain (prepopulate) the semantic memory system.

        Args:
            semantic_knowledge: e.g., [["desk", "atlocation", "officeroom"],
                ["chair", "atlocation", "officeroom",
                ["officeroom", "north", "livingroom]]

        """
        self.semantic_knowledge = semantic_knowledge
        for triple in self.semantic_knowledge:
            assert len(triple) == 3
            if self.is_full:
                break
            mem = [*triple, {"strength": 1}]  # num_generalized = 1
            self.add(mem)


class MemorySystems:
    """Multiple memory systems class.

    This class puts the short-term and long-term memory systems together. By doing so,
    it also creates a working memory system, which is a combination of the short-term
    and partial long-term memory. The partial long-term memory is created by retrieving
    memories from the long-term memory system based on the number of hops.

    Attributes:
        short: short-term memory system
        long: long-term memory system
        working: working memory system. This is short-term + partial long-term memory
        qualifier_relations: relations that can be used as qualifiers

    """

    def __init__(
        self,
        short: ShortMemory,
        long: LongMemory,
    ) -> None:
        """Bundle memory systems.

        Args:
            short: short-term memory system
            long: long-term memory system

        """
        self.short = short
        self.long = long
        self.qualifier_relations = ["current_time", "timestamp", "strength"]

    def forget_all(self) -> None:
        """Forget everything in the memory systems."""
        self.short.forget_all()
        self.long.forget_all()

    def get_working_memory(self, working_num_hops: int | None = None) -> Memory:
        """Get the working memory system. This is short-term + partial long-term memory.

        Args:
            working_num_hops: number of hops to consider when fetching long-term
                memories

        Returns:
            working: Memory

        """
        if working_num_hops is not None:
            raise NotImplementedError(
                "Not implemented yet. Please set working_num_hops to None."
            )

        working = []
        working += self.short.entries
        working += self.long.entries
        working = merge_lists(working)

        working = Memory(len(working), working)

        return working
