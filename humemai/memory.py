"""Memory system classes."""

import logging
import os
import random
from pprint import pformat

from .utils import merge_lists

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


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
        logging.debug(f"instantiating a memory object with size {capacity} ...")

        self.entries = []
        self.capacity = capacity
        assert self.capacity >= 0
        self._frozen = False

        logging.debug(f"Memory systrem with size {capacity} instantiated!")

        if memories is not None:
            for mem in memories:
                check, error_msg = self.can_be_added(mem)
                if not check:
                    logging.warning(error_msg)
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
        logging.debug(f"Adding a new memory entry {mem} ...")
        self.entries.append(mem)
        logging.info(
            f"memory entry {mem} added. Now there are in total of "
            f"{len(self.entries)} memories!"
        )

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
        logging.debug(f"Forgetting {mem} ...")
        self.entries.remove(mem)
        logging.info(f"{mem} forgotten!")

    def forget_all(self) -> None:
        """Forget everything in the memory system!"""
        if self.capacity == 0:
            error_msg = "The memory system capacity is 0. Can't forget all."
            logging.warning(error_msg)
            raise ValueError(error_msg)

        if self.is_frozen:
            error_msg = "The memory system is frozen. Can't forget all. Unfreeze first."
            logging.warning(error_msg)
            raise ValueError(error_msg)

        else:
            logging.warning("EVERYTHING IN THE MEMORY SYSTEM WILL BE FORGOTTEN!")
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
        logging.warning("forgetting a memory uniformly at random ...")
        mem = random.choice(self.entries)
        self.forget(mem)

    def increase_capacity(self, increase: int) -> None:
        """Increase the capacity.

        Args:
            increase: the amount of entries to increase.

        """
        assert isinstance(increase, int) and (not self.is_frozen)
        logging.debug(f"Increasing the memory capacity by {increase} ...")
        self.capacity += increase
        logging.info(
            f"The memory capacity has been increased by {increase} and now it's "
            f"{self.capacity}!"
        )

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
        logging.debug(f"Decreasing the memory capacity by {decrease} ...")
        self.capacity -= decrease
        logging.info(
            f"The memory capacity has been decreased by {decrease} and now it's "
            f"{self.capacity}!"
        )

    def return_as_list(self) -> list[list]:
        """Return the memories as a list of lists.

        Returns:
            a list of lists

        """
        return self.entries

    def query_memory(self, mem_query: list) -> list[list]:
        """Query memory.

        Args:
            mem_query: a quadruple, where each element can be "?". e.g.,
                ["bob", "atlocation", "?", 42], ["?", "atlocation", "officeroom", 42]
                "?" is used to match any value.

        Returns:
            mems_found: a list of memories that match the query


        """
        assert len(mem_query) == 4
        mems_found = []
        for mem in self.entries:
            if (mem_query[0] == mem[0]) or (mem_query[0] == "?"):
                if (mem_query[1] == mem[1]) or (mem_query[1] == "?"):
                    if (mem_query[2] == mem[2]) or (mem_query[2] == "?"):
                        if (mem_query[3] == mem[3]) or (mem_query[3] == "?"):
                            mems_found.append(mem)

        return mems_found


class EpisodicMemory(Memory):
    """Episodic memory class.


    Attributes:
        type (str): episodic
        entries (list): list of memories. Every memory is a quadruple: [head, relation,
            tail, {"timestamp": []}]
    """

    def __init__(
        self,
        capacity: int,
        memories: list[list] | None = None,
    ) -> None:
        """Init an episodic memory system.

        Args:
            capacity: capacity of the memory system (i.e., number of entries)
            memories: memories that can already be added from the beginning, if None,
                then it's an empty memory system.

        """
        super().__init__(capacity, memories)
        self.type = "episodic"

    def can_be_added(self, mem: list) -> tuple[bool, str | None]:
        """Check if an episodic memory can be added to the system or not.

        Args:
            mem: A memory as a quadraple: [head, relation, tail, {"timestamp": []}]

        Returns:
            True or False error_msg

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

    def add(self, mem: list) -> None:
        """Append a memory to the episodic memory system.

        After adding, it'll sort (ascending) the memories based on the timestamps.

        Args:
            mem: An episodic memory as a quadraple: [head, relation, tail, {"timestamp":
            []}]

        """
        added = False
        # Check if a list with the same first three elements exists
        for mem_ in self.entries:
            if mem_[:3] == mem[:3]:
                # Merge the timestamp lists
                mem_[3]["timestamp"] = sorted(
                    set(mem_[3]["timestamp"] + mem[3]["timestamp"])
                )
                added = True

        if not added:
            super().add(mem)

        # Define a helper function to get the max timestamp from the dict
        def max_timestamp(sublist):
            return max(sublist[3]["timestamp"])

        # Sort the list based on the max timestamp, in ascending order
        self.entries.sort(key=max_timestamp)

    def get_oldest_memory(self) -> list:
        return self.get_first_memory()

    def get_latest_memory(self) -> list:
        return self.get_last_memory()

    def forget_oldest(self) -> None:
        """Forget the oldest entry in the memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        """
        logging.debug("forgetting the oldest memory (FIFO)...")

        mem = self.get_oldest_memory()
        self.forget(mem)

    def forget_latest(self) -> None:
        """Forget the oldest entry in the memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        """
        logging.debug("forgetting the oldest memory (FIFO)...")

        mem = self.get_latest_memory()
        self.forget(mem)


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
        logging.debug(f"Turning an observation {ob} into a short memory ...")

        mem = ob[:-1] + [{"current_time": ob[-1]}]

        logging.info(f"Observation {ob} is now a short-term memory {mem}")

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
        logging.debug(f"Turning a short memory {short} into an episodic memory ...")
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
        logging.debug(f"Turning a short memory {short} into a semantic memory ...")
        sem = short[:-1] + [{"strength": 1}]

        return sem


class SemanticMemory(Memory):
    """Semantic memory class."""

    def __init__(self, capacity: int, memories: list[list] | None = None) -> None:
        """Init a semantic memory system.

        Args:
            capacity: capacity of the memory system (i.e., number of entries)
            memories: memories that can already be added from the beginning, if None,
                then it's an empty memory system.

        """
        super().__init__(capacity, memories)
        self.type = "semantic"

    def can_be_added(self, mem: list) -> bool:
        """Checks if a memory can be added to the system or not.

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

    def add(self, mem: list) -> None:
        """Append a memory to the semantic memory system.

        After adding, it'll sort (ascending) the memories based on the number of
        generalized samples.

        Args:
            mem: A memory as a quadruple: [head, relation, tail, {"strength": int}]

        """
        added = False
        # Check if a list with the same first three elements exists
        for mem_ in self.entries:
            if mem_[:3] == mem[:3]:
                # Merge the timestamp lists
                mem_[3]["strength"] = mem_[3]["strength"] + mem[3]["strength"]
                added = True

        if not added:
            super().add(mem)

        # Define a helper function to get the max strength from the dict
        def get_strength(sublist):
            return sublist[3]["strength"]

        # Sort the list based on the strength, in ascending order
        self.entries.sort(key=get_strength)

    def pretrain_semantic(
        self,
        semantic_knowledge: list[list],
        return_remaining_space: bool = True,
        freeze: bool = True,
    ) -> int:
        """Pretrain (prepopulate) the semantic memory system.

        Args:
            semantic_knowledge: e.g., [["desk", "atlocation", "officeroom"],
                ["chair", "atlocation", "officeroom",
                ["officeroom", "north", "livingroom]]
            return_remaining_space: whether or not to return the remaining space from the
                semantic memory system.
            freeze: whether or not to freeze the semantic memory system or not.

        Returns:
            free_space: free space that was not used, if any, so that it can be added to
                the episodic memory system.

        """
        self.semantic_knowledge = semantic_knowledge
        for triple in self.semantic_knowledge:
            assert len(triple) == 3
            if self.is_full:
                break
            mem = [*triple, {"strength": 1}]  # num_generalized = 1
            logging.debug(f"adding a pretrained semantic knowledge {mem}")
            self.add(mem)

        if return_remaining_space:
            free_space = self.capacity - len(self.entries)
            self.decrease_capacity(free_space)
            logging.info(
                f"The remaining space {free_space} will be returned. Now "
                f"the capacity of the semantic memory system is {self.capacity}"
            )

        else:
            free_space = None

        if freeze:
            self.freeze()
            logging.info("The semantic memory system is frozen!")

        return free_space

    def get_weakest_memory(self) -> list:
        return self.get_first_memory()

    def get_strongest_memory(self) -> list:
        return self.get_last_memory()

    def forget_weakest(self) -> None:
        """Forget the weakest entry in the semantic memory system.

        At the moment, this is simply done by looking up the number of generalized
        memories and comparing them.

        """
        logging.debug("forgetting the weakest memory ...")
        mem = self.get_weakest_memory()
        self.forget(mem)
        logging.info(f"{mem} is forgotten!")

    def forget_strongest(self) -> None:
        """Forget the strongest entry in the semantic memory system."""
        logging.debug("forgetting the strongest memory ...")
        mem = self.get_strongest_memory()
        self.forget(mem)
        logging.info(f"{mem} is forgotten!")


class WorkingMemory:
    """Working memory class."""

    def __init__(
        self,
        short: ShortMemory | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
        working_num_hops: int | None = None,
    ) -> None:
        """Working memory system.

        Args:
            short: short-term memory system
            episodic: episodic memory system
            semantic: semantic memory system
            working_num_hops: number of hops to consider when fetching long-term
                memories.

        """
        self.short = short
        self.episodic = episodic
        self.semantic = semantic
        self.working_num_hops = working_num_hops
        self.update()

    def __repr__(self):
        self.update()
        return pformat(self.entries, indent=4, width=1)

    def __iter__(self):
        self.update()
        return iter(self.entries[:])

    def __len__(self):
        self.update()
        return len(self.entries)

    def query_memory(self, mem_query: list) -> list[list]:
        """Query memory.

        Args:
            mem_query: a quadruple, where each element can be "?". e.g.,
                ["bob", "atlocation", "?", 42], ["?", "atlocation", "officeroom", 42]
                "?" is used to match any value.

        Returns:
            mems_found: a list of memories that match the query


        """
        self.update()
        assert len(mem_query) == 4
        mems_found = []
        for mem in self.entries:
            if (mem_query[0] == mem[0]) or (mem_query[0] == "?"):
                if (mem_query[1] == mem[1]) or (mem_query[1] == "?"):
                    if (mem_query[2] == mem[2]) or (mem_query[2] == "?"):
                        if (mem_query[3] == mem[3]) or (mem_query[3] == "?"):
                            mems_found.append(mem)

        return mems_found

    def update(self) -> None:
        """Update the working memory system."""
        self.entries = []

        if self.short is not None:
            self.entries += self.short.entries
        if self.episodic is not None:
            if self.working_num_hops is not None:
                raise NotImplementedError(
                    "Not implemented yet. Please set working_num_hops to None."
                )
            else:
                self.entries += self.episodic.entries
        if self.semantic is not None:
            if self.working_num_hops is not None:
                raise NotImplementedError(
                    "Not implemented yet. Please set working_num_hops to None."
                )
            else:
                self.entries += self.semantic.entries

        self.entries = merge_lists(self.entries)

    def to_list(self) -> list[list]:
        """Return memory systems as a list of lists.

        Returns:
            A list of lists. To distinguish the memory types, the last element includes
            a relation qualifier key-value pair. At the moment, memories are nothing but
            python lists (lists of quadruples), which is a mutable object. So, deepcopy
            it, if you want to keep the original state. In the future, a more suitable
            python object will be used to represent the graph structure of the memories.

        """
        self.update()
        return self.entries


class MemorySystems:
    """Multiple memory systems class.

    Attributes:
        episodic: episodic memory system
        semantic: semantic memory system
        short: short-term memory system
        qualifier_relations: relations that can be used as qualifiers
        working_num_hops: number of hops to consider when fetching long-term memories

    """

    def __init__(
        self,
        short: ShortMemory = None,
        episodic: EpisodicMemory = None,
        semantic: SemanticMemory = None,
        working_num_hops: int | None = None,
    ) -> None:
        """Bundle memory systems.

        Args:
            episodic: episodic memory system
            semantic: semantic memory system
            short: short-term memory system
            working_num_hops: number of hops to consider when fetching long-term
                memories

        """
        self.qualifier_relations = []
        self.working_num_hops = working_num_hops
        kwargs = {}
        if short is not None and short.capacity > 0:
            self.short = short
            self.qualifier_relations.append("current_time")
            kwargs["short"] = self.short

        if episodic is not None and episodic.capacity > 0:
            self.episodic = episodic
            self.qualifier_relations.append("timestamp")
            kwargs["episodic"] = self.episodic

        if semantic is not None and semantic.capacity > 0:
            self.semantic = semantic
            self.qualifier_relations.append("strength")
            kwargs["semantic"] = self.semantic

        kwargs["working_num_hops"] = self.working_num_hops
        self.working = WorkingMemory(**kwargs)

    def forget_all(self) -> None:
        """Forget everything in the memory systems."""
        if hasattr(self, "episodic"):
            self.episodic.forget_all()
        if hasattr(self, "semantic"):
            self.semantic.forget_all()
        if hasattr(self, "short"):
            self.short.forget_all()
