"""Memory system classes."""

import logging
import os
import random
from pprint import pformat

import numpy as np

from .utils import list_duplicates_of, remove_posession, remove_timestamp

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

    def __init__(self, capacity: int, memories: list | None = None) -> None:
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
                self.add(mem)

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

    def can_be_added(self, mem: list[str]) -> tuple[bool, str]:
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

        if self.is_full:
            return False, "The memory system is full!"

        return True, ""

    def __add__(self, other):
        entries = self.entries + other.entries
        return Memory(self.capacity + other.capacity, entries)

    def add(self, mem: list[str]) -> None:
        """Add memory to the memory system.

        Args:
           mem: A memory as a quadraple: [head, relation, tail, num]

        """
        check, error_msg = self.can_be_added(mem)
        if not check:
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Adding a new memory entry {mem} ...")
        self.entries.append(mem)
        logging.info(
            f"memory entry {mem} added. Now there are in total of "
            f"{len(self.entries)} memories!"
        )

        # sort ascending
        self.entries.sort(key=lambda x: x[-1])

    def can_be_forgotten(self, mem: list[str]) -> tuple[bool, str]:
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
                either a timestamp or num generalized, for episodic / short and semantic,
                respectively.

        """
        check, error_msg = self.can_be_forgotten(mem)
        if not check:
            logging.error(error_msg)
            raise ValueError(error_msg)

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

    def answer_random(self, query: list) -> tuple[str, int]:
        """Answer the question with a uniform-randomly chosen memory.

        Args:
            query: e.g., ["bob", "atlocation", "?", 42],
                ["?", "atlocation", "officeroom", 42], ["bob", "?", "officeroom", 42]

        Returns:
            pred: prediction (e.g., desk)
            num: either a timestamp or num generalized, for episodic / short and semantic,

        """
        if self.is_empty:
            logging.warning("Memory is empty. I can't answer any questions!")
            pred = None
            num = None

        else:
            mem = random.choice(self.entries)
            pred_idx = query.index("?")
            pred = mem[pred_idx]
            num = mem[-1]

        logging.info(f"pred: {pred}, num: {num}")

        return pred, num

    def answer_with_smallest_num(self, query: list[str]) -> tuple[str, int]:
        """Answer the question with the memory with the smallest num.

        Args:
            query: a quadruple, e.g., ["bob", "atlocation", "?", 42],
                ["?", "atlocation", "officeroom", 42]

        Returns:
            pred
            num

        """
        candidates = self.find_memory(query[:-1] + ["?"])

        if len(candidates) == 0:
            logging.info("no relevant memories found.")
            pred = None
            num = None
        else:
            logging.info(
                f"{len(candidates)} relevant memories were found in the entries!"
            )
            candidates.sort(key=lambda x: x[-1])
            candidate = candidates[0]
            query_idx = query.index("?")
            pred = candidate[query_idx]
            num = candidate[-1]

        return pred, num

    def answer_with_largest_num(self, query: list[str]) -> tuple[str, int]:
        """Answer the question with the memory with the largest num.

        Args:
            query: a quadruple, e.g., ["bob", "atlocation", "?", 42],
                ["?", "atlocation", "officeroom", 42]

        Returns:
            pred
            num

        """
        candidates = self.find_memory(query[:-1] + ["?"])

        if len(candidates) == 0:
            logging.info("no relevant memories found.")
            pred = None
            num = None
        else:
            logging.info(
                f"{len(candidates)} relevant memories were found in the entries!"
            )
            candidates.sort(key=lambda x: x[-1])
            candidate = candidates[-1]
            query_idx = query.index("?")
            pred = candidate[query_idx]
            num = candidate[-1]

        return pred, num

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
        """
        Return the memories as a list of lists.
        """
        return self.entries

    def find_memory(self, mem_query: list) -> list:
        """Find memory.

        Args:
            head: head
            relation: relation
            tail: tail
            num: a number

        Returns:
            mem: A memory as a quadraple: [head, relation, tail, num]

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
    """Episodic memory class."""

    def __init__(
        self,
        capacity: int,
        memories: list | None = None,
        remove_duplicates: bool = False,
    ) -> None:
        """Init an episodic memory system.

        Args:
            capacity: capacity of the memory system (i.e., number of entries)
            memories: memories that can already be added from the beginning, if None,
                then it's an empty memory system.
            remove_duplicates: if True, it'll remove the same memories with the older
                timestamps.

        """
        super().__init__(capacity, memories)
        self.type = "episodic"
        self.remove_duplicates = remove_duplicates

    def add(self, mem: list[str]) -> None:
        """Append a memory to the episodic memory system.

        Args:
            mem: An episodic memory as a quadraple: [head, relation, tail, timestamp]

        """
        super().add(mem)
        if self.remove_duplicates:
            self.clean_old_memories()

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

    def answer_oldest(self, query: list[str]) -> tuple[str, int]:
        """Answer the question with the oldest relevant memory.

        Args:
            query: a quadruple, e.g., ["bob", "atlocation", "?", 42],
                ["?", "atlocation", "officeroom", 42]

        Returns:
            pred: prediction
            timestamp: timestamp

        """
        return self.answer_with_smallest_num(query)

    def answer_latest(self, query: list[str]) -> tuple[str, int]:
        """Answer the question with the latest relevant memory.

        Args:
            query: a quadruple, e.g., ["bob", "atlocation", "?", 42],
                ["?", "atlocation", "officeroom", 42]

        Returns:
            pred: prediction
            timestamp: timestamp

        """
        return self.answer_with_largest_num(query)

    @staticmethod
    def ob2epi(ob: list[str]) -> list[str]:
        """Turn an observation into an episodic memory.

        At the moment, the observation format is the same as an episodic memory
        for simplification.

        Args:
            ob: An observation as a quadruple: [head, relation, tail, timestamp]

        Returns:
            mem: An episodic memory as a quadruple: [head, relation, tail, timestamp]

        """
        logging.debug(f"Turning an observation {ob} into a episodic memory ...")

        mem = ob

        logging.info(f"Observation {ob} is now a episodic memory {mem}")

        return mem

    def clean_old_memories(self) -> list:
        """Find if there are duplicate memories with different timestamps."""
        logging.debug("finding if duplicate memories exist ...")

        MARKER = "^^^"  # to allow hashing.
        entries = [MARKER.join(target[:-1]) for target in self.entries]

        logging.debug(f"There are {len(entries)} episodic memories before cleaning")
        uniques = set(entries)

        locs_all = [
            list_duplicates_of(entries, unique_entry) for unique_entry in uniques
        ]
        locs_all.sort(key=len)
        entries_cleaned = []

        for locs in locs_all:
            mem = self.entries[locs[0]]
            mem[-1] = max([self.entries[loc][-1] for loc in locs])
            entries_cleaned.append(mem)

        self.entries = entries_cleaned
        self.entries.sort(key=lambda x: x[-1])

        logging.debug(f"There are {len(self.entries)} episodic memories after cleaning")

    def find_similar_memories(self, split_possessive: bool = True) -> list:
        """Find N episodic memories that can be compressed into one semantic.

        At the moment, this is simply done by matching string values. If there are more
        than one group of similar episodic memories, it'll return the one with the
        largest number of memories.

        Args:
            split_possessive: whether to split the possessive, i.e., 's, or not.

        Returns:
            episodic_memories: similar episodic memories
            semantic_memory: encoded (compressed) semantic memory in a quadruple format
                (i.e., (head, relation, tail, num_generalized_memories))

        """
        logging.debug("looking for episodic entries that can be compressed ...")
        MARKER = "^^^"  # to allow hashing.

        semantic_possibles = [
            [e for e in remove_timestamp(entry)] for entry in self.entries
        ]

        if split_possessive:
            semantic_possibles = [
                [remove_posession(e) for e in entry] for entry in semantic_possibles
            ]

        semantic_possibles = [MARKER.join(elem) for elem in semantic_possibles]

        def duplicates(mylist, item):
            return [i for i, x in enumerate(mylist) if x == item]

        semantic_possibles = dict(
            (x, duplicates(semantic_possibles, x)) for x in set(semantic_possibles)
        )

        if len(semantic_possibles) == len(self.entries):
            logging.info("no episodic memories found to be compressible.")
            return None, None

        elif len(semantic_possibles) < len(self.entries):
            logging.debug("some episodic memories found to be compressible.")

            lens = [len(foo) for foo in list(semantic_possibles.values())]
            selected = np.argwhere(lens == np.max(lens)).flatten().tolist()

            max_keys = [list(semantic_possibles.keys())[i] for i in selected]
            max_key = random.choice(max_keys)  # if there is more than one

            indexes = semantic_possibles[max_key]

            episodic_memories = map(self.entries.__getitem__, indexes)
            episodic_memories = list(episodic_memories)
            # sort from the oldest to the latest
            episodic_memories = sorted(episodic_memories, key=lambda x: x[-1])
            semantic_memory = max_key.split(MARKER)
            # num_generalized_memories is the number of compressed episodic memories.
            semantic_memory.append(len(indexes))
            assert (len(semantic_memory)) == 4
            for mem in episodic_memories:
                assert len(mem) == 4

            logging.info(
                f"{len(indexes)} episodic memories can be compressed "
                f"into one semantic memory: {semantic_memory}."
            )
            return episodic_memories, semantic_memory
        else:
            raise ValueError("Something is wrong!")


class ShortMemory(Memory):
    """Short-term memory class."""

    def __init__(self, capacity: int, memories: list | None = None) -> None:
        super().__init__(capacity, memories)
        self.type = "short"

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

    @staticmethod
    def ob2short(ob: list[str]) -> list[str]:
        """Turn an observation into an short memory.

        At the moment, the observation format is almost the same as an episodic memory
        for simplification.

        Args:
            ob: An observation as a quadruple: [head, relation, tail, timestamp]

        Returns:
            mem: A short-term memory as a quadruple: [head, relation, tail, timestamp]

        """
        logging.debug(f"Turning an observation {ob} into a short memory ...")

        mem = ob

        logging.info(f"Observation {ob} is now a short-term memory {mem}")

        return mem

    @staticmethod
    def short2epi(short: list[str]) -> list[str]:
        """Turn a short memory into a episodic memory.

        Args:
            short: A short memory as a quadruple: [head, relation, tail, timestamp]

        Returns:
            epi: An episodic memory as a quadruple: [head, relation, tail, timestamp]

        """
        epi = short
        return epi

    @staticmethod
    def short2sem(short: list, split_possessive: bool = True) -> list:
        """Turn a short memory into a semantic memory.

        Args:
            short: A short memory as a quadruple: [head, relation, tail, timestamp]
            split_possessive: whether to split the possessive, i.e., 's, or not.

        Returns:
            sem: A semantic memory as a quadruple: [head, relation, tail,
                NUM_GENERALIZED]

        """
        sem = short[:-1]

        if split_possessive:
            sem = [remove_posession(elem) for elem in sem]

        sem += [1]

        return sem


class SemanticMemory(Memory):
    """Semantic memory class."""

    def __init__(self, capacity: int, memories: list | None = None) -> None:
        """Init a semantic memory system.

        Args:
            capacity: capacity of the memory system (i.e., number of entries)
            memories: memories that can already be added from the beginning, if None,
                then it's an empty memory system.

        """
        super().__init__(capacity, memories)
        self.type = "semantic"

    def can_be_added(self, mem: list[str]) -> bool:
        """Checks if a memory can be added to the system or not.

        Args:
            mem: A semantic memory as a quadraple: [head, relation, tail,
                num_generalized]

        Returns:
            True or False

        """
        if self.capacity == 0:
            return False, "The memory system capacity is 0!"

        if self._frozen:
            return False, "The memory system is frozen!"

        if self.is_full:
            if len(self.find_memory(mem[:-1] + ["?"])) == 0:
                return False, "The memory system is full!"
            else:
                return True, ""
        else:
            return True, ""

    def add(self, mem: dict):
        """Append a memory to the semantic memory system.

        Args:
            mem: A memory as a quadruple: [head, relation, tail, num_generalized]

        """
        super().add(mem)
        self.clean_same_memories()

    def pretrain_semantic(
        self,
        semantic_knowledge: list[list[str]],
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
            mem = [*triple, 1]  # num_generalized = 1
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

    def answer_weakest(
        self, query: list, split_possessive: bool = True
    ) -> tuple[str, int]:
        """Answer the question with the strongest relevant memory.

        Args:
            query: a quadruple, e.g., ["bob", "atlocation", "?", 42],
                ["?", "atlocation", "officeroom", 42]

        Returns:
            pred: prediction
            num_generalized: number of generalized samples.

        """
        logging.debug("answering a question with the answer_strongest policy ...")

        if split_possessive:
            query = [remove_posession(e) for e in query[:-1]] + [query[-1]]

        return self.answer_with_smallest_num(query)

    def answer_strongest(
        self, query: list, split_possessive: bool = True
    ) -> tuple[str, int]:
        """Answer the question with the strongest relevant memory.

        Args:
            query: a quadruple, e.g., ["bob", "atlocation", "?", 42],
                ["?", "atlocation", "officeroom", 42]

        Returns:
            pred: prediction
            num_generalized: number of generalized samples.

        """
        logging.debug("answering a question with the answer_strongest policy ...")

        if split_possessive:
            query = [remove_posession(e) for e in query[:-1]] + [query[-1]]

        return self.answer_with_largest_num(query)

    @staticmethod
    def ob2sem(ob: list, split_possessive: bool = True) -> dict:
        """Turn an observation into a semantic memory.

        At the moment, this is simply done by removing the names from the head and the
        tail.

        Args:
            ob: An observation as a quadruple: [head, relation, tail, timestamp]
            split_possessive: whether to split the possessive, i.e., 's, or not.

        Returns:
            mem: A semantic memory as a quadruple: [head, relation, tail, timestamp]

        """

        assert len(ob) == 4
        logging.debug(f"Turning an observation {ob} into a semantic memory ...")
        # split to remove the name
        if split_possessive:
            head, relation, tail = (
                remove_posession(ob[0]),
                remove_posession(ob[1]),
                remove_posession(ob[2]),
            )
        else:
            head, relation, tail = ob[0], ob[1], ob[2]

        # 1 stands for the 1 generalized.
        mem = [head, relation, tail, 1]
        logging.info(f"Observation {ob} is now a semantic memory {mem}")

        return mem

    def clean_same_memories(self) -> list:
        """Find if there are duplicate memories cuz they should be summed out.

        At the moment, this is simply done by matching string values.

        """
        logging.debug("finding if duplicate memories exist ...")

        entries = ["".join(target[:-1]) for target in self.entries]

        logging.debug(f"There are {len(entries)} semantic memories before cleaning")

        entries = ["".join(mem) for mem in entries]  # to make list hashable
        uniques = set(entries)

        locs_all = [
            list_duplicates_of(entries, unique_entry) for unique_entry in uniques
        ]
        locs_all.sort(key=len)
        entries_cleaned = []

        for locs in locs_all:
            mem = self.entries[locs[0]]
            mem[-1] = sum([self.entries[loc][-1] for loc in locs])
            entries_cleaned.append(mem)

        self.entries = entries_cleaned
        self.entries.sort(key=lambda x: x[-1])
        logging.debug(f"There are {len(self.entries)} episodic memories after cleaning")


class WorkingMemory(Memory):
    """Working memory class."""

    def __init__(self, capacity: int, memories: list | None = None) -> None:
        super().__init__(capacity)
        self.type = "working"


class MemorySystems:
    """Multiple memory systems class."""

    def __init__(
        self,
        episodic: EpisodicMemory = None,
        episodic_agent: EpisodicMemory = None,
        semantic: SemanticMemory = None,
        semantic_map: SemanticMemory = None,
        short: ShortMemory = None,
    ) -> None:
        """Bundle memory systems.

        Args:
            episodic: episodic memory system
            episodic_agent: episodic memory system for agent-related memories
            semantic: semantic memory system
            semantic_map: semantic memory system for the map
            short: short-term memory system

        """
        if episodic is not None and episodic.capacity > 0:
            self.episodic = episodic
        if episodic_agent is not None and episodic_agent.capacity > 0:
            self.episodic_agent = episodic_agent
        if semantic is not None and semantic.capacity > 0:
            self.semantic = semantic
        if semantic_map is not None and semantic_map.capacity > 0:
            self.semantic_map = semantic_map
        if short is not None and short.capacity > 0:
            self.short = short

    def get_working_memory(self, num_hops: int | str = "all") -> Memory:
        """Get the working memory system.

        The working memory system is defined as short-term memory + partial long-term
            memory (episodic and semantic).

        Args:
            num_hops: number of hops to consider when fetching long-term memories
            (episodic and semantic). If "all", then basically it's infinity, which
            means all long-term memories will be considered.

        Returns:
            working_memory: the working memory system

        """
        if num_hops == "all":
            return self.short + self.episodic + self.semantic
        else:
            raise NotImplementedError("Not implemented yet!")

    def return_as_a_dict_list(self) -> dict[str, list[list[str]]]:
        """Return memory systems as a dictionary of lists.

        Returns:
            to_return: a dictionary of lists. At the moment, memories are nothing but
                python lists (lists of quadruples), which is a mutable object. So,
                deepcopy it, if you want to keep the original state. In the future, a
                more suitable python object will be used to represent the graph
                structure of the memories.

        """
        to_return = {}
        if hasattr(self, "episodic"):
            to_return["episodic"] = self.episodic.return_as_list()
        if hasattr(self, "episodic_agent"):
            to_return["episodic_agent"] = self.episodic_agent.return_as_list()
        if hasattr(self, "semantic"):
            to_return["semantic"] = self.semantic.return_as_list()
        if hasattr(self, "semantic_map"):
            to_return["semantic_map"] = self.semantic_map.return_as_list()
        if hasattr(self, "short"):
            to_return["short"] = self.short.return_as_list()

        return to_return

    def forget_all(self) -> None:
        """Forget everything in the memory systems."""
        if hasattr(self, "episodic"):
            self.episodic.forget_all()
        if hasattr(self, "episodic_agent"):
            self.episodic_agent.forget_all()
        if hasattr(self, "semantic"):
            self.semantic.forget_all()
        if hasattr(self, "semantic_map"):
            self.semantic_map.forget_all()
        if hasattr(self, "short"):
            self.short.forget_all()

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)
