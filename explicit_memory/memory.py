"""Memory system classes."""
import logging
import os
import random
from copy import deepcopy
from pprint import pformat
from typing import Dict, List, Tuple, Union

import numpy as np

from .utils import list_duplicates_of, remove_posession, remove_timestamp

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Memory:
    """Memory (episodic, semantic, or short) class"""

    def __init__(self, memory_type: str, capacity: int) -> None:
        """

        Args
        ----
        memory_type: episodic, semantic, or short
        capacity: memory capacity

        """
        logging.debug(
            f"instantiating a {memory_type} memory object with size {capacity} ..."
        )

        assert memory_type in ["episodic", "semantic", "short"]
        self.type = memory_type
        self.entries = []
        self.capacity = capacity
        self._frozen = False

        logging.debug(f"{memory_type} memory object with size {capacity} instantiated!")

    def __repr__(self):
        return pformat(vars(self), indent=4, width=1)

    def forget(self, mem: List) -> None:
        """forget the given memory.

        Args
        ----
        mem: A memory as a quadraple: [head, relation, tail, num], where `num` is
            either a timestamp or num generalized, for episodic / short and semantic,
            respectively.

        """
        if self._frozen:
            error_msg = "The memory system is frozen!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        if mem not in self.entries:
            error_msg = f"{mem} is not in the memory system!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Forgetting {mem} ...")
        self.entries.remove(mem)
        logging.info(f"{mem} forgotten!")

    def forget_all(self) -> None:
        """Forget everything in the memory system!"""
        if self.is_frozen:
            error_msg = "The memory system is frozen. Can't forget all. Unfreeze first."
            logging.warning(error_msg)
            raise ValueError(error_msg)
        else:
            logging.warning("EVERYTHING IN THE MEMORY SYSTEM WILL BE FORGOTTEN!")
            self.entries = []

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

        Args
        ----
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

        Args
        ----
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

    def return_as_lists(self) -> List[list]:
        """
        Return the memories as a list of lists.
        """
        return deepcopy(self.entries)

    def find_memory(self, head: str, relation: str, tail: str) -> List[List[str]]:
        """Find memory.

        At least one of the two should be ?

        Args
        ----
        head: head
        relation: relation
        tail: tail

        Returns
        -------
        mem: A memory as a quadraple: [head, relation, tail, num]

        """
        assert (head != "?") or (relation != "?") or (tail != "?")
        mems_found = []
        for mem in self.entries:
            if (head == mem[0]) or (head == "?"):
                if (relation == mem[1]) or (relation == "?"):
                    if (tail == mem[2]) or (tail == "?"):
                        mems_found.append(mem)

        return mems_found


class EpisodicMemory(Memory):
    """Episodic memory class."""

    def __init__(self, capacity: int, remove_duplicates: bool = False) -> None:
        """Init an episodic memory system.

        Args
        ----
        capacity: capacity of the memory system (i.e., number of entries)
        remove_duplicates: if True, it'll remove the same memories with the older
            timestamps.

        """
        super().__init__("episodic", capacity)
        self.remove_duplicates = remove_duplicates

    def can_be_added(self) -> bool:
        """Checks if a memory can be added to the system or not.

        Returns
        -------
        True or False

        """
        if (self.capacity <= 0) or (self._frozen) or (self.is_full):
            return False

        else:
            return True

    def add(self, mem: list) -> None:
        """Append a memory to the episodic memory system.

        Args
        ----
        mem: An episodic memory as a quadraple: [head, relation, tail, timestamp]

        """
        if self._frozen:
            error_msg = "The memory system is frozen!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Adding a new memory entry {mem} ...")
        self.entries.append(mem)
        logging.info(
            f"memory entry {mem} added. Now there are in total of "
            f"{len(self.entries)} memories!"
        )
        if self.remove_duplicates:
            self.clean_old_memories()

        # sort ascending
        self.entries.sort(key=lambda x: x[-1])

        assert self.size <= self.capacity

    def get_oldest_memory(self) -> List:
        """Get the oldest memory in the episodic memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        Returns
        -------
        mem: the oldest memory as a quadraple

        """
        # sorted() is ascending by default.
        mem_candidate = sorted(self.entries, key=lambda x: x[-1])[0]
        mem = mem_candidate
        # mem = random.choice(
        #     [mem for mem in self.entries if mem_candidate[-1] == mem[-1]]
        # )
        logging.info(f"{mem} is the oldest memory in the system.")

        return mem

    def get_latest_memory(self) -> list:
        """Get the latest memory in the episodic memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them. If there are more than one memory with the same timestamp, then it'll
        choose one of them uniformly at random.

        Returns
        -------
        mem: An episodic memory as a quadraple: [head, relation, tail, timestamp]

        """
        # sorted() is ascending by default.
        mem_candidate = sorted(self.entries, key=lambda x: x[-1])[-1]
        mem = random.choice(
            [mem for mem in self.entries if mem_candidate[-1] == mem[-1]]
        )
        logging.info(f"{mem} is the oldest memory in the system.")

        return mem

    def forget_oldest(self) -> None:
        """Forget the oldest entry in the memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        """
        logging.debug("forgetting the oldest memory (FIFO)...")

        mem = self.get_oldest_memory()
        self.forget(mem)

    def answer_random(self, query: List) -> Tuple[str, int]:
        """Answer the question with a uniform-randomly chosen memory.

        Args
        ----
        query: e.g., ["bob", "atlocation", "?", 42],
            ["?", "atlocation", "officeroom", 42], ["bob", "?", "officeroom", 42]

        Returns
        -------
        pred: prediction (e.g., desk)
        timestamp

        """
        if self.is_empty:
            logging.warning("Memory is empty. I can't answer any questions!")
            pred = None
            timestamp = None

        else:
            mem = random.choice(self.entries)
            pred_idx = query.index("?")
            pred = mem[pred_idx]
            timestamp = mem[-1]

        logging.info(f"pred: {pred}, timestamp: {timestamp}")

        return pred, timestamp

    def answer_latest(self, query: List) -> Tuple[str, int]:
        """Answer the question with the latest relevant memory.

        If object X was found at Y and then later on found Z, then this strategy answers
        Z, instead of Y.

        Args
        ----
        query: e.g., ["bob", "atlocation", "?", 42],
            ["?", "atlocation", "officeroom", 42], ["bob", "?", "officeroom", 42]

        Returns
        -------
        pred: prediction
        timestamp: timestamp

        """
        logging.debug("answering a question with the answer_latest policy ...")

        if self.is_empty:
            logging.warning("Memory is empty. I can't answer any questions!")
            pred = None
            timestamp = None

        candidates = []

        for target in self.entries:
            assert len(query) == len(target) == 4
            count = 0
            for s, t in zip(query[:-1], target[:-1]):
                if s == t:
                    count += 1
            if count == 2:
                candidates.append(target)

        if len(candidates) == 0:
            logging.info("no relevant memories found.")
            pred = None
            timestamp = None
        else:
            logging.info(
                f"{len(candidates)} relevant memories were found in the entries!"
            )
            candidates.sort(key=lambda x: x[-1])
            candidate = candidates[-1]
            pred_idx = query.index("?")
            pred = candidate[pred_idx]
            timestamp = candidate[-1]

        logging.info(f"pred: {pred}, timestamp: {timestamp}")

        return pred, timestamp

    @staticmethod
    def ob2epi(ob: list) -> dict:
        """Turn an observation into an episodic memory.

        At the moment, the observation format is the same as an episodic memory
        for simplification.

        Args
        ----
        ob: An observation as a quadruple: [head, relation, tail, timestamp]

        Returns
        -------
        mem: An episodic memory as a quadruple: [head, relation, tail, timestamp]


        """
        logging.debug(f"Turning an observation {ob} into a episodic memory ...")

        mem = deepcopy(ob)

        logging.info(f"Observation {ob} is now a episodic memory {mem}")

        return mem

    def clean_old_memories(self) -> List:
        """Find if there are duplicate memories with different timestamps."""
        logging.debug("finding if duplicate memories exist ...")

        entries = ["".join(target[:-1]) for target in self.entries]

        logging.debug(f"There are {len(entries)} episodic memories before cleaning")

        entries = ["".join(mem) for mem in entries]  # to make list hashable
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
        logging.debug(f"There are {len(self.entries)} episodic memories after cleaning")

    def find_similar_memories(
        self, split_possessive: bool = True, dont_generalize_agent: bool = True
    ) -> List:
        """Find N episodic memories that can be compressed into one semantic.

        At the moment, this is simply done by matching string values. If there are more
        than one group of similar episodic memories, it'll return the one with the
        largest number of memories.

        Args
        ----
        split_possessive: whether to split the possessive, i.e., 's, or not.
        dont_generalize_agent: if True, the agent-related memories are not generalized,
            i.e., they are not put into the semantic memory system.


        Returns
        -------
        episodic_memories: similar episodic memories
        semantic_memory: encoded (compressed) semantic memory in a quadruple format
            (i.e., (head, relation, tail, num_generalized_memories))

        """
        logging.debug("looking for episodic entries that can be compressed ...")
        MARKER = "^^^"  # to allow hashing.

        if split_possessive:
            semantic_possibles = [
                [remove_posession(e) for e in remove_timestamp(entry)]
                for entry in self.entries
            ]
        else:
            semantic_possibles = [
                [e for e in remove_timestamp(entry)] for entry in self.entries
            ]
        if dont_generalize_agent:
            semantic_possibles = [
                entry
                for entry in semantic_possibles
                if entry[0] != "agent" and entry[1] != "agent" and entry[2] != "agent"
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

    def __init__(self, capacity: int) -> None:
        super().__init__("short", capacity)

    def add(self, mem: list) -> None:
        """Append a memory to the short memory system.

        Args
        ----
        mem: An episodic memory as a quadraple: [head, relation, tail, timestamp]

        """
        if self._frozen:
            error_msg = "The memory system is frozen!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        assert not self.is_full

        logging.debug(f"Adding a new memory entry {mem} ...")
        self.entries.append(mem)
        logging.info(
            f"memory entry {mem} added. Now there are in total of "
            f"{len(self.entries)} memories!"
        )
        # sort ascending
        self.entries.sort(key=lambda x: x[-1])

        assert self.size <= self.capacity

    def get_oldest_memory(self) -> list:
        """Get the oldest memory in the short-term memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        Returns
        -------
        mem: the oldest memory as a quadraple

        """
        # sorted() is ascending by default.
        mem_candidate = sorted(self.entries, key=lambda x: x[-1])[0]
        mem = mem_candidate
        # mem = random.choice(
        #     [mem for mem in self.entries if mem_candidate[-1] == mem[-1]]
        # )
        logging.info(f"{mem} is the oldest memory in the system.")

        return mem

    def get_latest_memory(self) -> list:
        """Get the latest memory in the short-term memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them. If there are more than one memory with the same timestamp, then it'll
        choose one of them uniformly at random.

        Returns
        -------
        mem: An episodic memory as a quadraple

        """
        # sorted() is ascending by default.
        mem_candidate = sorted(self.entries, key=lambda x: x[-1])[-1]
        mem = random.choice(
            [mem for mem in self.entries if mem_candidate[-1] == mem[-1]]
        )
        logging.info(f"{mem} is the oldest memory in the system.")

        return mem

    def forget_oldest(self) -> None:
        """Forget the oldest entry in the memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        """
        logging.debug("forgetting the oldest memory (FIFO)...")

        mem = self.get_oldest_memory()
        self.forget(mem)

    @staticmethod
    def ob2short(ob: list) -> list:
        """Turn an observation into an short memory.

        At the moment, the observation format is almost the same as an episodic memory
        for simplification.

        Args
        ----
        ob: An observation as a quadruple: [head, relation, tail, timestamp]

        Returns
        -------
        mem: A short-term memory as a quadruple: [head, relation, tail, timestamp]

        """
        logging.debug(f"Turning an observation {ob} into a short memory ...")

        mem = deepcopy(ob)

        logging.info(f"Observation {ob} is now a short-term memory {mem}")

        return mem

    @staticmethod
    def short2epi(short: list) -> list:
        """Turn a short memory into a episodic memory.

        Args
        ----
        short: A short memory as a quadruple: [head, relation, tail, timestamp]

        Returns
        -------
        epi: An episodic memory as a quadruple: [head, relation, tail, timestamp]
        """
        epi = deepcopy(short)
        return epi

    @staticmethod
    def short2sem(short: list, split_possessive: bool = True) -> list:
        """Turn a short memory into a episodic memory.

        Args
        ----
        short: A short memory as a quadruple: [head, relation, tail, timestamp]
        split_possessive: whether to split the possessive, i.e., 's, or not.

        Returns
        -------
        sem: A semantic memory as a quadruple: [head, relation, tail, NUM_GENERALIZED]


        """
        sem = deepcopy(short)

        if split_possessive:
            sem[0] = remove_posession(sem[0])
        else:
            sem[0] = sem[0]
        sem[-1] = 1

        return sem


class SemanticMemory(Memory):
    """Semantic memory class."""

    def __init__(
        self,
        capacity: int,
    ) -> None:
        """Init a semantic memory system.

        Args
        ----
        capacity: capacity of the memory system (i.e., number of entries)

        """
        super().__init__("semantic", capacity)

    def can_be_added(self, mem: List[List[str]]) -> bool:
        """Checks if a memory can be added to the system or not.

        Args
        ----
        True or False

        """
        if self.capacity <= 0:
            return False

        if self._frozen:
            return False

        if self.is_full:
            if self.find_same_memory(mem) is None:
                return False
            else:
                return True
        else:
            return True

    def pretrain_semantic(
        self,
        semantic_knowledge: list,
        return_remaining_space: bool = True,
        freeze: bool = True,
    ) -> int:
        """Pretrain (prepopulate) the semantic memory system.

        Args
        ----
        semantic_knowledge: e.g., [["desk", "atlocation", "officeroom"],
            ["chair", "atlocation", "officeroom",
            ["officeroom", "north", "livingroom]]
        return_remaining_space: whether or not to return the remaining space from the
            semantic memory system.
        freeze: whether or not to freeze the semantic memory system or not.

        Returns
        -------
        free_space: free space that was not used, if any, so that it can be added to
            the episodic memory system.
        """
        self.semantic_knowledge = deepcopy(semantic_knowledge)
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

    def get_weakest_memory(self) -> List:
        """Get the weakest memory in the semantic memory system system.

        At the moment, this is simply done by looking up num_generalized and comparing
        them. If there are more than one memory with the same num_generalized, then
        it'll choose one of them uniformly at random.

        Returns
        -------
        mem: the weakest memory as a quadraple

        """
        # sorted() is ascending by default.
        mem_candidate = sorted(self.entries, key=lambda x: x[-1])[0]
        mem = random.choice(
            [mem for mem in self.entries if mem_candidate[-1] == mem[-1]]
        )
        logging.info(f"{mem} is the weakest memory in the system.")

        return mem

    def get_strongest_memory(self) -> List:
        """Get the strongest memory in the semantic memory system system.

        At the moment, this is simply done by looking up num_generalized and comparing
        them. If there are more than one memory with the same num_generalized, then
        it'll choose one of them uniformly at random.

        Returns
        -------
        mem: the strongest memory as a quadraple

        """
        # sorted() is ascending by default.
        mem_candidate = sorted(self.entries, key=lambda x: x[-1])[-1]
        mem = random.choice(
            [mem for mem in self.entries if mem_candidate[-1] == mem[-1]]
        )
        logging.info(f"{mem} is the strongest memory in the system.")

        return mem

    def forget_weakest(self) -> None:
        """Forget the weakest entry in the semantic memory system.

        At the moment, this is simply done by looking up the number of generalized
        memories and comparing them.

        """
        logging.debug("forgetting the weakest memory ...")
        mem = self.get_weakest_memory()
        self.forget(mem)
        logging.info(f"{mem} is forgotten!")

    def answer_random(self, query: List) -> Tuple[str, int]:
        """Answer the question with a uniform-randomly chosen memory.

        Args
        ----
        query: e.g., ["bob", "atlocation", "?", 42],
            ["?", "atlocation", "officeroom", 42], ["bob", "?", "officeroom", 42]

        Returns
        -------
        pred: prediction (e.g., desk)
        num_generalized

        """
        if self.is_empty:
            logging.warning("Memory is empty. I can't answer any questions!")
            pred = None
            num_generalized = None

        else:
            mem = random.choice(self.entries)
            pred_idx = query.index("?")
            pred = mem[pred_idx]
            num_generalized = mem[-1]

        logging.info(f"pred: {pred}, num_generalized: {num_generalized}")

        return pred, num_generalized

    def answer_strongest(
        self, query: List, split_possessive: bool = True
    ) -> Tuple[str, int]:
        """Answer the question with the strongest relevant memory.

        Args
        ----
        query: e.g., ["bob", "atlocation", "?", 42],
            ["?", "atlocation", "officeroom", 42], ["bob", "?", "officeroom", 42]

        Returns
        -------
        pred: prediction
        num_generalized: number of generalized samples.

        """
        logging.debug("answering a question with the answer_strongest policy ...")

        if self.is_empty:
            logging.warning("Memory is empty. I can't answer any questions!")
            pred = None
            num_generalized = None

        candidates = []

        for target in self.entries:
            assert len(query) == len(target) == 4
            count = 0
            for s, t in zip(query[:-1], target[:-1]):
                if split_possessive:
                    if remove_posession(s) == t:
                        count += 1
                else:
                    if s == t:
                        count += 1
            if count == 2:
                candidates.append(target)

        if len(candidates) == 0:
            logging.info("no relevant memories found.")
            pred = None
            num_generalized = None
        else:
            logging.info(
                f"{len(candidates)} relevant memories were found in the entries!"
            )
            candidates.sort(key=lambda x: x[-1])
            candidate = candidates[-1]
            pred_idx = query.index("?")
            pred = candidate[pred_idx]
            num_generalized = candidate[-1]

        logging.info(f"pred: {pred}, num_generalized: {num_generalized}")

        return pred, num_generalized

    @staticmethod
    def ob2sem(ob: list, split_possessive: bool = True) -> dict:
        """Turn an observation into a semantic memory.

        At the moment, this is simply done by removing the names from the head and the
        tail.

        Args
        ----
        ob: An observation as a quadruple: [head, relation, tail, timestamp]
        split_possessive: whether to split the possessive, i.e., 's, or not.

        Returns
        -------
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

    def clean_same_memories(self) -> List:
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
        logging.debug(f"There are {len(self.entries)} episodic memories after cleaning")

    def add(self, mem: dict):
        """Append a memory to the semantic memory system.

        Args
        ----
        mem: A memory as a quadruple: [head, relation, tail, num_generalized]

        """
        if self._frozen:
            error_msg = "The memory system is frozen!"
            logging.error(error_msg)
            raise ValueError(error_msg)

        logging.debug(f"Adding a new memory entry {mem} ...")
        self.entries.append(mem)
        logging.info(
            f"memory entry {mem} added. Now there are in total of "
            f"{len(self.entries)} memories!"
        )
        self.clean_same_memories()

        # sort ascending
        self.entries.sort(key=lambda x: x[-1])

        assert self.size <= self.capacity

    def find_same_memory(self, mem: List[List[str]]) -> List[List[str]]:
        """Find a semantic memory that's the same as the query memory.

        Args
        ----
        mem: A semantic memory in a quadruple: [head, relation, tail, num_generalized]

        Returns
        -------
        A semantic memory. If it doesn't exist, then return None.

        """
        candidate = [entry for entry in self.entries if mem[:-1] == entry[:-1]]

        if len(candidate) == 0:
            return None

        elif len(candidate) == 1:
            return candidate[0]
        else:
            raise ValueError("Something is wrong!")
