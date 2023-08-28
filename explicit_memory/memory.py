"""Memory system classes."""
import logging
import os
import random
from copy import deepcopy
from pprint import pformat
from typing import List, Tuple

from .utils import get_duplicate_dicts, list_duplicates_of

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

    def forget(self, mem: dict) -> None:
        """forget the given memory.

        Args
        ----
        mem: A memory in a dictionary format.

            for episodic and short:
            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

            for semantic:
            {"object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "num_generalized": <NUM_GENERALIZED>}

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
            logging.warning(
                "The memory system is frozen. Can't forget all. Unfreeze first."
            )
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
        """Forget a memory in the memory system in a uniform distribution manner."""
        logging.warning("forgetting a random memory using a uniform distribution ...")
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

    def return_as_dicts(self) -> dict:
        """
        Return the memories as dicts, not as the Memory object.
        """
        return deepcopy(self.entries)


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

    def can_be_added(self, mem: dict) -> bool:
        """Checks if a memory can be added to the system or not.

        Args
        ----
        mem: An episodic memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        Returns
        -------
        True or False

        """
        if (self.capacity <= 0) or (self._frozen) or (self.is_full):
            return False

        else:
            return True

    def add(self, mem: dict) -> None:
        """Append a memory to the episodic memory system.

        Args
        ----
        mem: An episodic memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

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
        self.entries.sort(key=lambda x: x["timestamp"])

        assert self.size <= self.capacity

    def get_oldest_memory(self, entries: list = None) -> List:
        """Get the oldest memory in the episodic memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        Returns
        -------
        mem: the oldest memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # sorted() is ascending by default.
        mem_candidate = sorted(entries, key=lambda x: x["timestamp"])[0]
        mem = random.choice(
            [mem for mem in entries if mem_candidate["timestamp"] == mem["timestamp"]]
        )
        logging.info(f"{mem} is the oldest memory in the entries.")

        return mem

    def get_latest_memory(self, entries: list = None) -> dict:
        """Get the latest memory in the episodic memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        Returns
        -------
        mem: the latest memory in a dictionary format

            for episodic:
            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # sorted() is ascending by default.
        mem_candidate = sorted(entries, key=lambda x: x["timestamp"])[-1]
        mem = random.choice(
            [mem for mem in entries if mem_candidate["timestamp"] == mem["timestamp"]]
        )
        logging.info(f"{mem} is the oldest memory in the entries.")

        return mem

    def forget_oldest(self) -> None:
        """Forget the oldest entry in the memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        """
        logging.debug("forgetting the oldest memory (FIFO)...")

        mem = self.get_oldest_memory()
        self.forget(mem)

    def answer_random(self) -> Tuple[str, int]:
        """Answer the question with a uniform-randomly chosen memory.

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
            pred = mem["object_location"]
            timestamp = mem["timestamp"]

        logging.info(f"pred: {pred}, timestamp: {timestamp}")

        return pred, timestamp

    def answer_latest(self, question: dict) -> Tuple[str, int]:
        """Answer the question with the latest relevant memory.

        If object X was found at Y and then later on found Z, then this strategy answers
        Z, instead of Y.

        Args
        ----
        question: a dict (i.e., {"human": <HUMAN>, "object": <OBJECT>})

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

        else:
            duplicates = get_duplicate_dicts(
                {"human": question["human"], "object": question["object"]}, self.entries
            )

            if len(duplicates) == 0:
                logging.info("no relevant memories found.")
                pred = None
                timestamp = None

            else:
                logging.info(
                    f"{len(duplicates)} relevant memories were found in the entries!"
                )
                mem = self.get_latest_memory(duplicates)
                pred = mem["object_location"]
                timestamp = mem["timestamp"]

        logging.info(f"pred: {pred}, timestamp: {timestamp}")

        return pred, timestamp

    @staticmethod
    def ob2epi(ob: dict) -> dict:
        """Turn an observation into an episodic memory.

        At the moment, the observation format is the same as an episodic memory
        for simplification.

        Args
        ----
        ob: An observation in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>,
            "object_location": <OBJECT_LOCATION>, "current_time": <CURRENT_TIME>}

        Returns
        -------
        mem: An episodic memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        logging.debug(f"Turning an observation {ob} into a episodic memory ...")

        mem = deepcopy(ob)
        mem["timestamp"] = mem.pop("current_time")

        logging.info(f"Observation {ob} is now a episodic memory {mem}")

        return mem

    def find_same_memory(self, mem) -> dict:
        """Find an episodic memory that's almost the same as the query memory.

        Args
        ----
        mem: An episodic memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>,
             "object_location": <OBJECT_LOCATION>, "timestamp": <TIMESTAMP>}

        Returns
        -------
        an episodic memory if it exists. Otherwise return None.

        """
        for entry in self.entries:
            if (
                (entry["human"] == mem["human"])
                and (entry["object"] == mem["object"])
                and (entry["object_location"] == mem["object_location"])
            ):
                return entry

        return None

    def clean_old_memories(self) -> List:
        """Find if there are duplicate memories with different timestamps."""
        logging.debug("finding if duplicate memories exist ...")

        entries = deepcopy(self.entries)
        logging.debug(f"There are {len(entries)} episdoic memories before cleaning")
        for entry in entries:
            del entry["timestamp"]

        entries = [str(mem) for mem in entries]  # to make list hashable
        uniques = set(entries)

        locs_all = [
            list_duplicates_of(entries, unique_entry) for unique_entry in uniques
        ]
        locs_all.sort(key=len)
        entries_cleaned = []

        for locs in locs_all:
            mem = self.entries[locs[0]]
            mem["timestamp"] = max([self.entries[loc]["timestamp"] for loc in locs])
            entries_cleaned.append(mem)

        self.entries = entries_cleaned
        logging.debug(f"There are {len(self.entries)} episdoic memories after cleaning")


class ShortMemory(Memory):
    """Short-term memory class."""

    def __init__(self, capacity: int) -> None:
        super().__init__("short", capacity)

    def add(self, mem: dict) -> None:
        """Append a memory to the short memory system.

        mem: A short memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

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
        self.entries.sort(key=lambda x: x["timestamp"])

        assert self.size <= self.capacity

    def get_oldest_memory(self, entries: list = None) -> List:
        """Get the oldest memory in the episodic memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        Returns
        -------
        mem: the oldest memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # sorted() is ascending by default.
        mem_candidate = sorted(entries, key=lambda x: x["timestamp"])[0]
        mem = random.choice(
            [mem for mem in entries if mem_candidate["timestamp"] == mem["timestamp"]]
        )
        logging.info(f"{mem} is the oldest memory in the entries.")

        return mem

    def get_latest_memory(self, entries: list = None) -> dict:
        """Get the latest memory in the episodic memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        Returns
        -------
        mem: the latest memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # sorted() is ascending by default.
        mem_candidate = sorted(entries, key=lambda x: x["timestamp"])[-1]
        mem = random.choice(
            [mem for mem in entries if mem_candidate["timestamp"] == mem["timestamp"]]
        )
        logging.info(f"{mem} is the oldest memory in the entries.")

        return mem

    def forget_oldest(self) -> None:
        """Forget the oldest entry in the memory system.

        At the moment, this is simply done by looking up the timestamps and comparing
        them.

        """
        logging.debug("forgetting the oldest memory (FIFO)...")

        mem = self.get_oldest_memory()
        self.forget(mem)

    def find_similar_memories(self, mem) -> None:
        """Find similar memories.

        mem: A short memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        logging.debug("Searching for similar memories in the short memory system...")
        similar = []
        for entry in self.entries:
            if (entry["object"] == mem["object"]) and (
                entry["object_location"] == mem["object_location"]
            ):
                similar.append(entry)

        logging.info(f"{len(similar)} similar short memories found!")

        return similar

    @staticmethod
    def ob2short(ob: dict) -> dict:
        """Turn an observation into an short memory.

        At the moment, the observation format is almost the same as an episodic memory
        for simplification.

        Args
        ----
        ob: An observation in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>,
            "object_location": <OBJECT_LOCATION>, "current_time": <CURRENT_TIME>}

        Returns
        -------
        mem: An episodic memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        logging.debug(f"Turning an observation {ob} into a short memory ...")

        mem = deepcopy(ob)
        mem["timestamp"] = mem.pop("current_time")

        logging.info(f"Observation {ob} is now a short-term memory {mem}")

        return mem

    @staticmethod
    def short2epi(short: dict) -> dict:
        """Turn a short memory into a episodic memory.

        Args
        ----
        short: A short memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        Returns
        -------
        epi: An episodic memory in a dictionary format

            {"human": <HUMAN>,
             "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        epi = deepcopy(short)
        return epi

    @staticmethod
    def short2sem(short: dict) -> dict:
        """Turn a short memory into a episodic memory.

        Args
        ----
        short: A short memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        Returns
        -------
        sem: A semantic memory in a dictionary format

            {"object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "num_generalized": <NUM_GENERALIZED>}

        """
        sem = deepcopy(short)

        del sem["human"]
        del sem["timestamp"]
        sem["num_generalized"] = 1

        return sem

    def find_same_memory(self, mem) -> dict:
        """Find a short memory that's almost the same as the query memory.

        Args
        ----
        mem: A short memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>,
             "object_location": <OBJECT_LOCATION>, "timestamp": <TIMESTAMP>}

        Returns
        -------
        A short memory if it exists. Otherwise return None.

        """
        for entry in self.entries:
            if (
                (entry["human"] == mem["human"])
                and (entry["object"] == mem["object"])
                and (entry["object_location"] == mem["object_location"])
            ):
                return entry

        return None


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

    def can_be_added(self, mem: dict) -> bool:
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
        semantic_knowledge: dict,
        return_remaining_space: bool = True,
        freeze: bool = True,
    ) -> int:
        """Pretrain the semantic memory system from ConceptNet.

        Args
        ----
        semantic_knowledge: from ConceptNet.
        return_remaining_space: whether or not to return the remaining space from the
            semantic memory system.
        freeze: whether or not to freeze the semantic memory system or not.

        Returns
        -------
        free_space: free space that was not used, if any, so that it can be added to
            the episodic memory system.
        """
        self.semantic_knowledge = deepcopy(semantic_knowledge)
        for obj, loc in self.semantic_knowledge.items():
            if self.is_full:
                break
            mem = {"object": obj, "object_location": loc, "num_generalized": 1}
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

    def get_weakest_memory(self, entries: list = None) -> List:
        """Get the weakest memory in the semantic memory system system.

        At the moment, this is simply done by looking up the number of generalized
        memories comparing them. In the end, an RL agent has to learn this
        by itself.

        Returns
        -------
        mem: the weakest memory in a dictionary format

            for semantic:
            {"object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "num_generalized": <NUM_GENERALIZED>}

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # sorted() is ascending by default.
        mem_candidate = sorted(entries, key=lambda x: x["num_generalized"])[0]
        mem = random.choice(
            [
                mem
                for mem in entries
                if mem_candidate["num_generalized"] == mem["num_generalized"]
            ]
        )
        logging.info(f"{mem} is the weakest memory in the entries.")

        return mem

    def get_strongest_memory(self, entries: list = None) -> List:
        """Get the strongest memory in the semantic memory system system.

        At the moment, this is simply done by looking up the number of generalized
        memories comparing them.

        Returns
        -------
        mem: the strongest memory in a dictionary format


            for semantic:
            {"object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "num_generalized": <NUM_GENERALIZED>}

        """
        if entries is None:
            logging.debug("No entries were specified. We'll use the memory system.")
            entries = self.entries

        # sorted() is ascending by default.
        mem_candidate = sorted(entries, key=lambda x: x["num_generalized"])[-1]
        mem = random.choice(
            [
                mem
                for mem in entries
                if mem_candidate["num_generalized"] == mem["num_generalized"]
            ]
        )
        logging.info(f"{mem} is the strongest memory in the entries.")

        return mem

    def find_similar_memories(self, mem) -> None:
        """Find similar memories.

        mem: A short memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "timestamp": <TIMESTAMP>}

        """
        logging.debug("Searching for similar memories in the short memory system...")
        similar = []
        for entry in self.entries:
            if (entry["object"] == mem["object"]) and (
                entry["object_location"] == mem["object_location"]
            ):
                similar.append(entry)

        logging.info(f"{len(similar)} similar short memories found!")

        return similar

    def forget_weakest(self) -> None:
        """Forget the weakest entry in the semantic memory system.

        At the moment, this is simply done by looking up the number of generalized
        memories and comparing them.

        """
        logging.debug("forgetting the weakest memory ...")
        mem = self.get_weakest_memory()
        self.forget(mem)
        logging.info(f"{mem} is forgotten!")

    def answer_random(self) -> Tuple[str, int]:
        """Answer the question with a uniform-randomly chosen memory.

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
            pred = mem["object_location"]
            num_generalized = mem["num_generalized"]

        logging.info(f"pred: {pred}, num_generalized: {num_generalized}")

        return pred, num_generalized

    def answer_strongest(self, question: list) -> Tuple[str, int]:
        """Answer the question (Find the head that matches the question, and choose the
        strongest one among them).

        Args
        ----
        question: a dict (i.e., {"human": <HUMAN>, "object": <OBJECT>})

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

        else:
            duplicates = get_duplicate_dicts(
                {"object": question["object"]}, self.entries
            )
            if len(duplicates) == 0:
                logging.info("no relevant memories found.")
                pred = None
                num_generalized = None

            else:
                logging.info(
                    f"{len(duplicates)} relevant memories were found in the entries!"
                )
                mem = self.get_strongest_memory(duplicates)
                pred = mem["object_location"]
                num_generalized = mem["num_generalized"]

        logging.info(f"pred: {pred}, num_generalized: {num_generalized}")

        return pred, num_generalized

    @staticmethod
    def ob2sem(ob: dict) -> dict:
        """Turn an observation into a semantic memory.

        At the moment, this is simply done by removing the names from the head and the
        tail.

        Args
        ----
        ob: An observation in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>,
            "object_location": <OBJECT_LOCATION>, "current_time": <CURRENT_TIME>}


        Returns
        -------
        mem: A semantic memory in a dictionary format

            {"human": <HUMAN>, "object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "num_generalized": <NUM_GENERALIZED>}

        """
        logging.debug(f"Turning an observation {ob} into a semantic memory ...")
        mem = deepcopy(ob)

        del mem["human"]
        del mem["current_time"]

        # 1 stands for the 1 generalized.
        mem["num_generalized"] = 1

        logging.info(f"Observation {ob} is now a semantic memory {mem}")

        return mem

    def clean_same_memories(self) -> List:
        """Find if there are duplicate memories cuz they should be summed out.

        At the moment, this is simply done by matching string values.

        """
        logging.debug("finding if duplicate memories exist ...")

        entries = deepcopy(self.entries)
        logging.debug(
            f"There are in total of {len(entries)} semantic memories before cleaning"
        )
        for entry in entries:
            del entry["num_generalized"]

        entries = [str(mem) for mem in entries]  # to make list hashable
        uniques = set(entries)

        locs_all = [
            list_duplicates_of(entries, unique_entry) for unique_entry in uniques
        ]
        locs_all.sort(key=len)
        entries_cleaned = []

        for locs in locs_all:
            mem = self.entries[locs[0]]
            mem["num_generalized"] = sum(
                [self.entries[loc]["num_generalized"] for loc in locs]
            )
            entries_cleaned.append(mem)

        self.entries = entries_cleaned
        logging.debug(
            f"There are now in total of {len(self.entries)} semantic memories after cleaning"
        )

    def add(self, mem: dict):
        """Append a memory to the semantic memory system.

        Args
        ----
        mem: A memory in a dictionary format

            {"object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "num_generalized": <NUM_GENERALIZED>}

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
        self.entries.sort(key=lambda x: x["num_generalized"])

        assert self.size <= self.capacity

    def find_same_memory(self, mem) -> dict:
        """Find a semantic memory that's almost the same as the query memory.

        Args
        ----
        mem: A semantic memory in a dictionary format

            {"object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "num_generalized": <NUM_GENERALIZED>}

        Returns
        -------
        A semantic memory if it exists. Otherwise return None.

        """
        for entry in self.entries:
            if (entry["object"] == mem["object"]) and (
                entry["object_location"] == mem["object_location"]
            ):
                return entry

        return None

    def find_same_object_memory(self, mem) -> dict:
        """Find a semantic memory whose object is the same as the query memory.

        Args
        ----
        mem: A semantic memory in a dictionary format

            {"object": <OBJECT>, "object_location": <OBJECT_LOCATION>,
             "num_generalized": <NUM_GENERALIZED>}

        Returns
        -------
        A semantic memory if it exists. Otherwise return None.

        """
        for entry in self.entries:
            if (entry["object"] == mem["object"]) and (
                entry["object_location"] != mem["object_location"]
            ):
                return entry

        return None
