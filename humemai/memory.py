"""Memory system classes."""

from typing import Literal
import random
from pprint import pformat

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

    def __init__(self, capacity: int, memories: list[list] | None = None) -> None:
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

        if memories is not None:
            for mem in memories:
                check, error_msg = self.can_be_added(mem)
                if not check:
                    raise ValueError(error_msg)
                else:
                    self.add(mem)

    def __repr__(self):
        return pformat(self.entries, indent=4, width=1)

    def __iter__(self):
        return iter(self.entries[:])

    def __len__(self):
        return len(self.entries)

    def __add__(self, other):
        entries = self.entries + other.entries
        return Memory(self.capacity + other.capacity, entries)

    def can_be_added(self, mem: list) -> tuple[bool, str | None]:
        """Check if a memory can be added to the system or not.

        Args:
            mem: A memory as a quadraple: [head, relation, tail, num]

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
        self.entries.append(mem)

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

    def get_first_memory(self) -> None:
        """Get the first memory in the memory system.

        Returns:
            mem: the first memory as a quadraple

        """
        return self.entries[0]

    def get_last_memory(self) -> None:
        """Get the last memory in the memory system.

        Returns:
            mem: the last memory as a quadraple

        """
        return self.entries[-1]

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

    def return_as_list(self) -> list[list]:
        """Return the memories as a list of lists.

        Returns:
            a list of lists

        """
        return self.entries


class ShortMemory(Memory):
    """Short-term memory class."""

    def __init__(self, capacity: int, memories: list[list] | None = None) -> None:
        super().__init__(capacity, memories)
        self.type = "short"

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
        self.type = "long"

    def fetch_by_selection(
        self, selection: Literal["oldest", "latest", "weakest", "strongest"]
    ) -> list:
        """Fetch a memory based on the selection.

        Args:
            selection: "oldest", "latest", "weakest", or "strongest"

        Returns:
            mem: a memory as a quadraple
        """
        if selection == "oldest":
            return sorted(self.entries, key=lambda x: max(x[-1]["timestamp"]))[0]
        elif selection == "latest":
            return sorted(self.entries, key=lambda x: max(x[-1]["timestamp"]))[-1]
        elif selection == "weakest":
            return sorted(self.entries, key=lambda x: x[-1]["strength"])[0]
        elif selection == "strongest":
            return sorted(self.entries, key=lambda x: x[-1]["strength"])[-1]
        else:
            raise ValueError(
                "`selection` should be 'oldest', 'latest', 'weakest', or 'strongest'"
            )

    def forget_by_selection(
        self, selection: Literal["oldest", "latest", "weakest", "strongest"]
    ) -> None:
        """Forget a memory based on the selection.

        Args:
            selection: "oldest", "latest", "weakest", or "strongest"

        """
        mem = self.fetch_by_selection(selection)
        self.forget(mem)

    def can_be_added_as_episodic(self, mem: list) -> tuple[bool, str | None]:
        """Check if a memory can be added as an episodic memory to the long-term
        memory system.

        Args:
            mem: An episodic memory as a quadraple: [head, relation, tail, {"timestamp":
            []}]

        Returns:
            True or False, error_msg

        """
        check, error_msg = super().can_be_added(mem)
        if not check:
            return check, error_msg

        if list(mem[-1].keys())[0] != "timestamp":
            return False, "The memory should have timestamp!"

        if self.is_full:
            for entry in self.entries:
                if entry[:-1] == mem[:-1]:
                    return True, None

            return False, "The memory system is full!"

        else:
            return True, None

    def can_be_added_as_semantic(self, mem: list) -> tuple[bool, str | None]:
        """Check if a memory can be added as a semantic memory to the long-term memory
        system.

        Args:
            mem: A semantic memory as a quadraple: [head, relation, tail,
                {"strength": int}]

        Returns:
            True or False, error_msg

        """
        check, error_msg = super().can_be_added(mem)
        if not check:
            return check, error_msg

        if list(mem[-1].keys())[0] != "strength":
            return False, "The memory should have strength!"

        if self.is_full:
            for entry in self.entries:
                if entry[:-1] == mem[:-1]:
                    return True, None

            return False, "The memory system is full!"

        else:
            return True, None

    def add_as_episodic(self, mem: list) -> None:
        """Append a memory as an episodic memory to the long-term memory system.

        Args:
            mem: An episodic memory as a quadraple: [head, relation, tail, {"timestamp":
            []}]

        """
        assert self.can_be_added_as_episodic(mem)[0]
        added = False
        # Check if a list with the same first three elements exists
        for entry in self.entries:
            if entry[:-1] == mem[:-1] and "timestamp" in entry[-1]:
                # Merge the timestamp lists
                entry[-1]["timestamp"] = sorted(
                    set(entry[-1]["timestamp"] + mem[-1]["timestamp"])
                )
                added = True

        if not added:
            super().add(mem)

    def add_as_semantic(self, mem: list) -> None:
        """Append a memory as a semantic memory to the long-term memory system.

        Args:
            mem: A semantic memory as a quadraple: [head, relation, tail,
                {"strength": int}]

        """
        assert self.can_be_added_as_semantic(mem)[0]
        added = False
        # Check if a list with the same first three elements exists
        for entry in self.entries:
            if entry[:-1] == mem[:-1] and "strength" in entry[-1]:
                # Merge the strength values
                entry[-1]["strength"] += mem[-1]["strength"]

                added = True

        if not added:
            super().add(mem)

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

    Attributes:
        short: short-term memory system
        long: long-term memory system
        working: working memory system. This is short-term + partial long-term memory
        qualifier_relations: relations that can be used as qualifiers
        working_num_hops: number of hops to consider when fetching long-term memories

    """

    def __init__(
        self,
        short: ShortMemory = None,
        long: LongMemory = None,
    ) -> None:
        """Bundle memory systems.

        Args:
            short: short-term memory system
            long: long-term memory system

        """
        self.qualifier_relations = ["current_time", "timestamp", "strength"]
        self.short = short
        self.long = long

    def forget_all(self) -> None:
        """Forget everything in the memory systems."""
        self.short.forget_all()
        self.long.forget_all()

    def get_working_memory(
        self,
        sort_by: Literal["current_time", "timestamp", "strength"] = "current_time",
        working_num_hops: int | None = None,
    ) -> list[list]:
        """Get the working memory system. This is short-term + partial long-term memory

        Args:
            sort_by: sort by "current_time", "timestamp", or "strength"
            working_num_hops: number of hops to consider when fetching long-term
                memories

        Returns:
            working: a list of memories

        """
        if working_num_hops is not None:
            raise NotImplementedError(
                "Not implemented yet. Please set working_num_hops to None."
            )

        working = []
        working += self.short.entries
        working += self.long.entries
        working = merge_lists(self.entries)

        if sort_by == "current_time":
            return working
        elif sort_by == "timestamp":
            return sorted(working, key=lambda x: max(x[-1]["timestamp"]))
        elif sort_by == "strength":
            return sorted(working, key=lambda x: x[-1]["strength"])
        else:
            raise ValueError(
                "`sort_by` should be 'current_time', 'timestamp', or 'strength'"
            )

    def query_working_memory(self, mem_query: list) -> list[list]:
        """Query memory.

        Args:
            mem_query: a quadruple, where each element can be "?". e.g.,
                ["bob", "atlocation", "?", "?], ["?", "atlocation", "officeroom", "?"]
                "?" is used to match any value.

        Returns:
            mems_found: a list of memories that match the query


        """
        assert len(mem_query) == 4
        mems_found = []

        for mem in self.get_working_memory():
            if (mem_query[0] == mem[0]) or (mem_query[0] == "?"):
                if (mem_query[1] == mem[1]) or (mem_query[1] == "?"):
                    if (mem_query[2] == mem[2]) or (mem_query[2] == "?"):
                        if (mem_query[3] == mem[3]) or (mem_query[3] == "?"):
                            mems_found.append(mem)

        return mems_found
