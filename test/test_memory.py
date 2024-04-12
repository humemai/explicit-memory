import itertools
import random
import unittest

from humemai.memory import *


class MemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = Memory(capacity=8)

    def test_can_be_added(self):
        memory = Memory(capacity=0)
        self.assertFalse(memory.can_be_added(["foo", "bar", "baz", 1])[0])

        memory = Memory(capacity=4)
        memory.freeze()
        self.assertFalse(memory.can_be_added(["foo", "bar", "baz", 1])[0])

        memory.unfreeze()
        memory = Memory(capacity=1)
        memory.add(["foo", "bar", "baz", 1])
        self.assertFalse(memory.can_be_added(["foo", "bar", "baz", 1])[0])

    def test_add(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.assertEqual(self.memory.size, 1)

        self.memory.freeze()
        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 1])

        self.memory.unfreeze()
        self.assertTrue(not self.memory.is_frozen)

        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 5])
        self.memory.add(["foo", "bar", "baz", 6])
        self.memory.add(["foo", "bar", "baz", 7])
        self.memory.add(["foo", "bar", "baz", 8])

        self.assertTrue(self.memory.is_full)

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 9])

    def test_can_be_forgotten(self):
        memory = Memory(capacity=0)
        check, error_msg = memory.can_be_forgotten(["foo", "bar", "baz", 1])
        self.assertFalse(check)

        memory = Memory(capacity=4)
        check, error_msg = memory.can_be_forgotten(["foo", "bar", "baz", 1])
        self.assertFalse(check)

        memory = Memory(capacity=4)
        memory.freeze()
        check, error_msg = memory.can_be_forgotten(["foo", "bar", "baz", 1])
        self.assertFalse(check)

        memory = Memory(capacity=2)
        memory.add(["foo", "bar", "baz", 1])
        check, error_msg = memory.can_be_forgotten(["foo", "bar", "qux", 1])
        self.assertFalse(check)

    def test_forget(self):
        with self.assertRaises(ValueError):
            self.memory.forget(["foo", "bar", "baz", 2])

        self.memory.freeze()
        self.assertTrue(self.memory.is_frozen)
        with self.assertRaises(ValueError):
            self.memory.forget(["foo", "bar", "baz", 1])

        self.memory.unfreeze()
        self.assertTrue(not self.memory.is_frozen)
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "foo", "foo", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        self.memory.forget(["foo", "bar", "baz", 1])
        self.assertEqual(self.memory.size, 2)

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

    def test_forget_all(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])

        self.memory.forget_all()
        self.assertEqual(self.memory.size, 0)
        self.assertTrue(self.memory.is_empty)

        memory = Memory(0)
        with self.assertRaises(ValueError):
            memory.forget_all()

    def test_forget_random(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])

        self.memory.forget_random()
        self.assertEqual(self.memory.size, 2)

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

    def test_increase_capacity(self):
        self.memory.increase_capacity(16)
        self.assertEqual(self.memory.capacity, 24)

    def test_decrease_capacity(self):
        self.memory.decrease_capacity(4)
        self.assertEqual(self.memory.capacity, 4)

    def test_return_as_list(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])

        returned = self.memory.return_as_list()
        self.assertEqual(len(returned), 3)
        del returned
        self.assertEqual(self.memory.size, 3)

        self.memory.forget_all()
        self.memory.add(["foo", "bar", "baz", 1])
        returned = self.memory.return_as_list()
        self.assertEqual(returned, [["foo", "bar", "baz", 1]])

        self.memory.forget_all()
        self.memory.add(["foo", "bar", "baz", 1])
        returned = self.memory.return_as_list()
        returned[-1] = 2
        self.assertNotEqual(returned, [["foo", "bar", "baz", 1]])

    def test_find_memory(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        with self.assertRaises(AssertionError):
            mems_found = self.memory.find_memory(["foo", "bar", "baz"])

        mems_found = self.memory.find_memory(["foo", "baz", "bar", "?"])
        self.assertEqual(len(mems_found), 0)

        mems_found = self.memory.find_memory(["foo", "bar", "?", "?"])
        self.assertEqual(len(mems_found), 3)

        mems_found = self.memory.find_memory(["?", "?", "?", "?"])
        self.assertEqual(len(mems_found), 4)

        mems_found = self.memory.find_memory(["?", "?", "?", 5])
        self.assertEqual(len(mems_found), 1)

        mems_found = self.memory.find_memory(["foo", "bar", "baz", 1])
        self.assertEqual(len(mems_found), 1)

        mems_found = self.memory.find_memory(["foo", "bar", "?", 4])
        self.assertEqual(len(mems_found), 0)

        mems_found = self.memory.find_memory(["foo", "bar", "baz", 1])
        # mems_found[0][0] = "qux"
        self.assertEqual(self.memory.entries[0], ["foo", "bar", "baz", 1])

    def test_answer_random(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])

        self.assertEqual(self.memory.answer_random(["foo", "bar", "?", 42])[0], "baz")

    def test_answer_with_smallest_num(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_with_smallest_num(["foo", "bar", "?", 42])
        self.assertEqual((pred, num), ("baz", 1))

        pred, num = self.memory.answer_with_smallest_num(["foo", "baz", "?", 42])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_with_smallest_num(["?", "bar", "baz", 42])
        self.assertEqual((pred, num), ("foo", 1))

        pred, num = self.memory.answer_with_smallest_num(["?", "foo", "bar", 42])
        self.assertEqual((pred, num), ("baz", 5))

    def test_answer_with_largest_num(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_with_largest_num(["foo", "bar", "?", 42])
        self.assertEqual((pred, num), ("baz", 3))

        pred, num = self.memory.answer_with_largest_num(["foo", "baz", "?", 42])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_with_largest_num(["?", "bar", "baz", 42])
        self.assertEqual((pred, num), ("foo", 3))

        pred, num = self.memory.answer_with_largest_num(["?", "foo", "bar", 42])
        self.assertEqual((pred, num), ("baz", 5))


class EpisodicMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = EpisodicMemory(capacity=8, remove_duplicates=False)

    def test_add(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.assertEqual(self.memory.size, 1)

        self.memory.freeze()
        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 1])

        self.memory.unfreeze()
        self.assertTrue(not self.memory.is_frozen)

        self.memory.add(["foo", "bar", "baz", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 7])
        self.memory.add(["foo", "bar", "baz", 5])
        self.memory.add(["foo", "bar", "baz", 6])
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 2])

        self.assertTrue(self.memory.is_full)

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 0])

    def test_get_oldest_memory(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        oldest = self.memory.get_oldest_memory()
        self.assertEqual(oldest, ["foo", "bar", "baz", 1])
        # oldest[-1] = 1234
        self.assertEqual(
            self.memory.entries,
            [
                ["foo", "bar", "baz", 1],
                ["foo", "bar", "baz", 3],
                ["foo", "bar", "baz", 4],
            ],
        )

    def test_get_latest_memory(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        latest = self.memory.get_latest_memory()
        self.assertEqual(latest, ["foo", "bar", "baz", 4])
        # latest[-1] = 1234
        self.assertEqual(
            self.memory.entries,
            [
                ["foo", "bar", "baz", 1],
                ["foo", "bar", "baz", 3],
                ["foo", "bar", "baz", 4],
            ],
        )

    def test_forget_oldest(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        self.memory.forget_oldest()
        self.assertEqual(self.memory.size, 2)

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

        self.assertEqual(
            self.memory.entries, [["foo", "bar", "baz", 3], ["foo", "bar", "baz", 4]]
        )

    def test_forget_latest(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        self.memory.forget_latest()
        self.assertEqual(self.memory.size, 2)

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

        self.assertEqual(
            self.memory.entries, [["foo", "bar", "baz", 1], ["foo", "bar", "baz", 3]]
        )

    def test_answer_oldest(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_oldest(["foo", "bar", "?", 42])
        self.assertEqual((pred, num), ("baz", 1))

        pred, num = self.memory.answer_oldest(["foo", "baz", "?", 42])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_oldest(["?", "bar", "baz", 42])
        self.assertEqual((pred, num), ("foo", 1))

        pred, num = self.memory.answer_oldest(["?", "foo", "bar", 42])
        self.assertEqual((pred, num), ("baz", 5))

    def test_answer_latest(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_latest(["foo", "bar", "?", 42])
        self.assertEqual((pred, num), ("baz", 3))

        pred, num = self.memory.answer_latest(["foo", "baz", "?", 42])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_latest(["?", "bar", "baz", 42])
        self.assertEqual((pred, num), ("foo", 3))

        pred, num = self.memory.answer_latest(["?", "foo", "bar", 42])
        self.assertEqual((pred, num), ("baz", 5))

    def test_ob2epi(self):
        ob = ["foo", "bar", "baz", 1]
        epi = EpisodicMemory.ob2epi(["foo", "bar", "baz", 1])

        self.assertEqual(ob, epi)

    def test_clean_old_memories(self):
        self.memory.add(["tae's foo", "bar", "baz", 3])
        self.memory.add(["tae's foo", "bar", "baz", 2])
        self.memory.add(["tae's foo", "bar", "baz", 2])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["baz", "foo", "bar", 10])
        self.memory.add(["qux", "foo", "bar", 6])

        self.memory.clean_old_memories()
        self.assertEqual(self.memory.size, 4)

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

        self.assertEqual(
            self.memory.entries,
            [
                ["baz", "qux", "bar", 1],
                ["tae's foo", "bar", "baz", 3],
                ["qux", "foo", "bar", 6],
                ["baz", "foo", "bar", 10],
            ],
        )

    def test_find_similar_memories(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=True
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))

        self.memory.forget_all()
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["agent", "qux", "bar", 1])
        self.memory.add(["baz", "qux", "agent", 1])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=True
        )
        self.assertEqual(
            episodic_memories,
            [
                ["foo", "bar", "baz", 1],
                ["foo", "bar", "baz", 2],
                ["foo", "bar", "baz", 3],
            ],
        )
        self.assertEqual(semantic_memory, ["foo", "bar", "baz", 3])

        self.memory.forget_all()
        self.memory.add(["tae's foo", "bar", "baz", 3])
        self.memory.add(["tae's foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["agent", "qux", "bar", 1])
        self.memory.add(["baz", "qux", "agent", 1])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )
        self.assertEqual(
            episodic_memories,
            [
                ["tae's foo", "bar", "baz", 2],
                ["tae's foo", "bar", "baz", 3],
            ],
        )
        self.assertEqual(semantic_memory, ["tae's foo", "bar", "baz", 2])

        self.memory.forget_all()
        self.memory.add(["tae's foo", "bar", "baz", 3])
        self.memory.add(["tae's foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["agent", "qux", "bar", 1])
        self.memory.add(["baz", "qux", "agent", 1])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=True
        )
        self.assertEqual(
            episodic_memories,
            [
                ["foo", "bar", "baz", 1],
                ["tae's foo", "bar", "baz", 2],
                ["tae's foo", "bar", "baz", 3],
            ],
        )
        self.assertEqual(semantic_memory, ["foo", "bar", "baz", 3])

        self.memory.forget_all()
        self.memory.add(["tae's foo", "bar", "baz", 3])
        self.memory.add(["tae's bar", "bar", "baz", 2])
        self.memory.add(["tae's", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["agent", "qux", "bar", 1])
        self.memory.add(["baz", "qux", "agent", 1])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=True
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))

        self.memory.forget_all()
        self.memory.add(["agent", "qux", "bar", 7])
        self.memory.add(["agent", "qux", "bar", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 6])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )
        self.assertEqual(
            episodic_memories,
            [
                ["agent", "qux", "bar", 5],
                ["agent", "qux", "bar", 6],
                ["agent", "qux", "bar", 7],
                ["agent", "qux", "bar", 8],
            ],
        )
        self.assertEqual(semantic_memory, ["agent", "qux", "bar", 4])

        self.memory.forget_all()
        self.memory.add(["agent", "qux", "bar", 7])
        self.memory.add(["agent", "qux", "bar", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 6])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )
        self.assertEqual(
            episodic_memories,
            [
                ["agent", "qux", "bar", 5],
                ["agent", "qux", "bar", 6],
                ["agent", "qux", "bar", 7],
                ["agent", "qux", "bar", 8],
            ],
        )
        self.assertEqual(semantic_memory, ["agent", "qux", "bar", 4])

        self.memory.forget_all()
        self.memory.add(["agent", "qux", "bar", 7])
        self.memory.add(["agent", "qux", "bar", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 5])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )

        if episodic_memories != [
            ["foo", "bar", "baz", 1],
            ["foo", "bar", "baz", 2],
            ["foo", "bar", "baz", 3],
        ]:
            self.assertEqual(
                episodic_memories,
                [
                    ["agent", "qux", "bar", 5],
                    ["agent", "qux", "bar", 7],
                    ["agent", "qux", "bar", 8],
                ],
            )
            self.assertEqual(semantic_memory, ["agent", "qux", "bar", 3])

        if episodic_memories != [
            ["agent", "qux", "bar", 5],
            ["agent", "qux", "bar", 7],
            ["agent", "qux", "bar", 8],
        ]:
            self.assertEqual(
                episodic_memories,
                [
                    ["foo", "bar", "baz", 1],
                    ["foo", "bar", "baz", 2],
                    ["foo", "bar", "baz", 3],
                ],
            )
            self.assertEqual(semantic_memory, ["foo", "bar", "baz", 3])


class EpisodicMemoryRemoveDuplicatesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = EpisodicMemory(capacity=8, remove_duplicates=True)

    def test_add(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.assertEqual(self.memory.size, 1)

        self.memory.freeze()
        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 1])

        self.memory.unfreeze()
        self.assertTrue(not self.memory.is_frozen)

        self.memory.add(["foo", "bar", "baz", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 7])
        self.memory.add(["foo", "bar", "baz", 5])
        self.memory.add(["foo", "bar", "baz", 6])
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 2])

        self.assertEqual(self.memory.size, 1)
        self.assertEqual(self.memory.get_oldest_memory(), ["foo", "bar", "baz", 8])
        self.assertEqual(self.memory.get_latest_memory(), ["foo", "bar", "baz", 8])

        for i in range(7):
            self.memory.add([str(i), str(i), str(i), 8 - i])

        self.assertTrue(self.memory.is_full)
        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

    def test_get_oldest_memory(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        oldest = self.memory.get_oldest_memory()
        self.assertEqual(oldest, ["foo", "bar", "baz", 4])
        # oldest[-1] = 1234
        self.assertEqual(
            self.memory.entries,
            [
                ["foo", "bar", "baz", 4],
            ],
        )

    def test_get_latest_memory(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        latest = self.memory.get_latest_memory()
        self.assertEqual(latest, ["foo", "bar", "baz", 4])
        # latest[-1] = 1234
        self.assertEqual(
            self.memory.entries,
            [
                ["foo", "bar", "baz", 4],
            ],
        )

    def test_forget_oldest(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        self.memory.forget_oldest()
        self.assertEqual(self.memory.size, 0)
        self.assertTrue(self.memory.is_empty)

    def test_forget_latest(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        self.memory.forget_latest()
        self.assertEqual(self.memory.size, 0)
        self.assertTrue(self.memory.is_empty)

    def test_answer_oldest(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_oldest(["foo", "bar", "?", 42])
        self.assertEqual((pred, num), ("baz", 3))

        pred, num = self.memory.answer_oldest(["foo", "baz", "?", 42])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_oldest(["?", "bar", "baz", 42])
        self.assertEqual((pred, num), ("foo", 3))

        pred, num = self.memory.answer_oldest(["?", "foo", "bar", 42])
        self.assertEqual((pred, num), ("baz", 5))

    def test_answer_latest(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_latest(["foo", "bar", "?", 42])
        self.assertEqual((pred, num), ("baz", 3))

        pred, num = self.memory.answer_latest(["foo", "baz", "?", 42])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_latest(["?", "bar", "baz", 42])
        self.assertEqual((pred, num), ("foo", 3))

        pred, num = self.memory.answer_latest(["?", "foo", "bar", 42])
        self.assertEqual((pred, num), ("baz", 5))

    def test_ob2epi(self):
        ob = ["foo", "bar", "baz", 1]
        epi = EpisodicMemory.ob2epi(["foo", "bar", "baz", 1])

        self.assertEqual(ob, epi)

    def test_clean_old_memories(self):
        self.memory.add(["tae's foo", "bar", "baz", 3])
        self.memory.add(["tae's foo", "bar", "baz", 2])
        self.memory.add(["tae's foo", "bar", "baz", 2])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["baz", "foo", "bar", 10])

        self.memory.clean_old_memories()
        self.assertEqual(self.memory.size, 3)

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

        self.assertEqual(
            self.memory.entries,
            [
                ["baz", "qux", "bar", 1],
                ["tae's foo", "bar", "baz", 3],
                ["baz", "foo", "bar", 10],
            ],
        )

    def test_find_similar_memories(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=True
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))

        self.memory.forget_all()
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["agent", "qux", "bar", 1])
        self.memory.add(["baz", "qux", "agent", 1])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=True
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))

        self.memory.forget_all()
        self.memory.add(["tae's foo", "bar", "baz", 3])
        self.memory.add(["tae's foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["agent", "qux", "bar", 1])
        self.memory.add(["baz", "qux", "agent", 1])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))

        self.memory.forget_all()
        self.memory.add(["tae's foo", "bar", "baz", 3])
        self.memory.add(["tae's foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["baz", "qux", "bar", 1])
        self.memory.add(["agent", "qux", "bar", 1])
        self.memory.add(["baz", "qux", "agent", 1])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=True
        )
        self.assertEqual(
            episodic_memories,
            [
                ["foo", "bar", "baz", 1],
                ["tae's foo", "bar", "baz", 3],
            ],
        )
        self.assertEqual(semantic_memory, ["foo", "bar", "baz", 2])

        self.memory.forget_all()
        self.memory.add(["agent", "qux", "bar", 7])
        self.memory.add(["agent", "qux", "bar", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 6])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))

        self.memory.forget_all()
        self.memory.add(["agent", "qux", "bar", 7])
        self.memory.add(["agent", "qux", "bar", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 6])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))

        self.memory.forget_all()
        self.memory.add(["agent", "qux", "bar", 7])
        self.memory.add(["agent", "qux", "bar", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["baz", "foo", "bar", 5])
        self.memory.add(["agent", "qux", "bar", 5])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))

        self.memory.forget_all()
        self.memory.add(["agent", "qux", "bar", 7])
        self.memory.add(["agent", "qux", "bar", 8])
        self.memory.add(["foo", "bar", "agent", 3])
        self.memory.add(["agent", "bar", "baz", 2])
        self.memory.add(["foo", "agent", "baz", 1])
        self.memory.add(["baz", "agent", "bar", 5])
        self.memory.add(["agent", "agent", "bar", 5])

        episodic_memories, semantic_memory = self.memory.find_similar_memories(
            split_possessive=False
        )
        self.assertEqual((episodic_memories, semantic_memory), (None, None))


class ShortMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = ShortMemory(capacity=8)

    def test_ob2short(self):
        ob = ["foo", "bar", "baz", 1]
        short = ShortMemory.ob2short(["foo", "bar", "baz", 1])

        self.assertEqual(ob, short)

    def test_short2epi(self):
        short = ["foo", "bar", "baz", 1]
        epi = ShortMemory.short2epi(["foo", "bar", "baz", 1])

        self.assertEqual(short, epi)

    def test_short2sem(self):
        short = ["foo", "bar", "baz", 4]
        sem = ShortMemory.short2sem(short, split_possessive=False)

        self.assertEqual(["foo", "bar", "baz", 1], sem)

        short = ["tae's foo", "bar", "tae's baz", 4]
        sem = ShortMemory.short2sem(short, split_possessive=True)

        self.assertEqual(["foo", "bar", "baz", 1], sem)


class SemanticMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = SemanticMemory(capacity=8)

    def test_can_be_added(self):
        memory = SemanticMemory(capacity=0)
        self.assertFalse(memory.can_be_added(["foo", "bar", "baz", 1])[0])

        memory = SemanticMemory(capacity=4)
        memory.freeze()
        self.assertFalse(memory.can_be_added(["foo", "bar", "baz", 1])[0])

        memory = SemanticMemory(capacity=1)
        memory.add(["foo", "bar", "baz", 1])
        self.assertTrue(memory.can_be_added(["foo", "bar", "baz", 1])[0])

    def test_add(self):
        self.memory.add(["foo", "bar", "baz", 2])
        self.assertEqual(self.memory.size, 1)

        self.memory.freeze()
        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 1])

        self.memory.unfreeze()
        self.memory.add(["foo", "bar", "baz", 2])
        self.assertEqual(self.memory.size, 1)

        self.memory.add(["baz", "bar", "foo", 1])
        self.assertEqual(self.memory.size, 2)

        nums = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(nums), nums)

        self.memory.add(["foo", "bar", "baz", 8])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "bar", "baz", 7])
        self.memory.add(["foo", "bar", "baz", 5])
        self.memory.add(["foo", "bar", "baz", 6])
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 2])

        self.assertEqual(self.memory.size, 2)
        self.assertEqual(self.memory.get_strongest_memory(), ["foo", "bar", "baz", 39])
        self.assertEqual(self.memory.get_weakest_memory(), ["baz", "bar", "foo", 1])

        for i in range(6):
            self.memory.add([str(i), str(i), str(i), random.randint(1, 10)])

        self.assertTrue(self.memory.is_full)
        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

        memory = SemanticMemory(capacity=2)
        memory.add(["foo", "bar", "baz", 3])
        memory.add(["foo", "bar", "baz", 1])
        self.assertEqual(memory.size, 1)
        memory.add(["foo", "bar", "qux", 1])
        self.assertEqual(memory.size, 2)
        memory.add(["foo", "bar", "qux", 4])
        self.assertEqual(memory.size, 2)
        with self.assertRaises(ValueError):
            memory.add(["baz", "bar", "qux", 5])

        timestamps = [mem[-1] for mem in memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)
        self.assertEqual(timestamps, [4, 5])

    def test_pretrain_semantic(self):
        semantic_knowledge = list(itertools.permutations(["foo", "bar", "baz"]))
        free_space = self.memory.pretrain_semantic(
            semantic_knowledge, return_remaining_space=True
        )
        self.assertEqual(free_space, 2)
        self.assertEqual(self.memory.size, 6)
        nums = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(nums, [1] * 6)

        self.memory.unfreeze()
        self.memory.forget_all()
        self.memory.increase_capacity(2)
        semantic_knowledge = list(
            itertools.permutations(["foo", "bar", "baz", "qux"], 3)
        )
        free_space = self.memory.pretrain_semantic(
            semantic_knowledge, return_remaining_space=False, freeze=False
        )
        self.assertEqual(free_space, None)
        self.assertEqual(self.memory.size, 8)
        nums = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(nums, [1] * 8)

        self.assertFalse(self.memory.is_frozen)

    def test_get_weakest_memory(self):
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "baz", "bar", 1])
        self.memory.add(["baz", "foo", "bar", 2])

        weakest = self.memory.get_weakest_memory()
        self.assertEqual(weakest, ["foo", "baz", "bar", 1])
        self.assertEqual(
            self.memory.entries,
            [
                ["foo", "baz", "bar", 1],
                ["baz", "foo", "bar", 2],
                ["foo", "bar", "baz", 3],
            ],
        )

    def test_get_strongest_memory(self):
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "baz", "bar", 1])
        self.memory.add(["baz", "foo", "bar", 2])

        strongest = self.memory.get_strongest_memory()
        self.assertEqual(strongest, ["foo", "bar", "baz", 3])
        self.assertEqual(
            self.memory.entries,
            [
                ["foo", "baz", "bar", 1],
                ["baz", "foo", "bar", 2],
                ["foo", "bar", "baz", 3],
            ],
        )

    def test_forget_weakest(self):
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "baz", "bar", 1])
        self.memory.add(["baz", "foo", "bar", 2])

        self.memory.forget_weakest()
        self.assertEqual(self.memory.size, 2)
        self.assertEqual(
            self.memory.entries,
            [
                ["baz", "foo", "bar", 2],
                ["foo", "bar", "baz", 3],
            ],
        )

    def test_forget_strongest(self):
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "baz", "bar", 1])
        self.memory.add(["baz", "foo", "bar", 2])

        self.memory.forget_strongest()
        self.assertEqual(self.memory.size, 2)
        self.assertEqual(
            self.memory.entries,
            [
                ["foo", "baz", "bar", 1],
                ["baz", "foo", "bar", 2],
            ],
        )

    def test_answer_weakest(self):
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "baz", "bar", 1])
        self.memory.add(["baz", "foo", "bar", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["qux", "bar", "baz", 1])

        pred, num = self.memory.answer_weakest(["foo", "bar", "?", 42])
        self.assertEqual((pred, num), ("baz", 4))

        pred, num = self.memory.answer_weakest(["foo", "baz", "?", 42])
        self.assertEqual((pred, num), ("bar", 1))

        pred, num = self.memory.answer_weakest(["?", "bar", "baz", 42])
        self.assertEqual((pred, num), ("qux", 1))

        pred, num = self.memory.answer_weakest(["?", "foo", "bar", 42])
        self.assertEqual((pred, num), ("baz", 2))

    def test_answer_strongest(self):
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["foo", "baz", "bar", 1])
        self.memory.add(["baz", "foo", "bar", 2])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["qux", "bar", "baz", 1])

        pred, num = self.memory.answer_strongest(["foo", "bar", "?", 42])
        self.assertEqual((pred, num), ("baz", 4))

        pred, num = self.memory.answer_strongest(["foo", "baz", "?", 42])
        self.assertEqual((pred, num), ("bar", 1))

        pred, num = self.memory.answer_strongest(["?", "bar", "baz", 42])
        self.assertEqual((pred, num), ("foo", 4))

        pred, num = self.memory.answer_strongest(["?", "foo", "bar", 42])
        self.assertEqual((pred, num), ("baz", 2))

    def test_ob2sem(self):
        sem = SemanticMemory.ob2sem(["foo", "bar", "baz", 3], split_possessive=True)
        self.assertEqual(sem, ["foo", "bar", "baz", 1])

        sem = SemanticMemory.ob2sem(["foo", "bar", "baz", 1], split_possessive=False)
        self.assertEqual(sem, ["foo", "bar", "baz", 1])

        sem = SemanticMemory.ob2sem(
            ["tae's foo", "bar", "tae's baz", 100], split_possessive=True
        )
        self.assertEqual(sem, ["foo", "bar", "baz", 1])

        sem = SemanticMemory.ob2sem(
            ["foo", "tae's bar", "baz", 100], split_possessive=False
        )
        self.assertEqual(sem, ["foo", "tae's bar", "baz", 1])

    def test_clean_same_memories(self):
        self.memory.entries = [
            ["foo", "bar", "baz", 3],
            ["foo", "baz", "bar", 1],
            ["baz", "foo", "bar", 2],
            ["foo", "bar", "baz", 1],
            ["qux", "bar", "baz", 1],
        ]

        self.memory.clean_same_memories()
        self.assertEqual(self.memory.size, 4)

        self.assertTrue(
            (
                self.memory.entries
                == [
                    ["qux", "bar", "baz", 1],
                    ["foo", "baz", "bar", 1],
                    ["baz", "foo", "bar", 2],
                    ["foo", "bar", "baz", 4],
                ]
            )
            or (
                self.memory.entries
                == [
                    ["foo", "baz", "bar", 1],
                    ["qux", "bar", "baz", 1],
                    ["baz", "foo", "bar", 2],
                    ["foo", "bar", "baz", 4],
                ]
            )
        )


class MemorySystemsTest(unittest.TestCase):
    def test_all(self) -> None:
        self.memory_systems = MemorySystems(
            episodic=EpisodicMemory(capacity=8),
            semantic=SemanticMemory(capacity=4),
            short=ShortMemory(capacity=1),
        )

        to_return = self.memory_systems.return_as_a_dict_list()
        self.assertTrue(to_return["episodic"] == [])
        self.assertTrue(to_return["semantic"] == [])
        self.assertTrue(to_return["short"] == [])
        self.assertTrue("episodic_agent" not in to_return)

        self.memory_systems.episodic.add(["foo", "bar", "baz", 1])
        self.memory_systems.semantic.add(["foo", "bar", "baz", 1])
        self.memory_systems.short.add(["foo", "bar", "baz", 1])

        self.memory_systems.forget_all()
        self.assertTrue(self.memory_systems.episodic.is_empty)
        self.assertTrue(self.memory_systems.semantic.is_empty)
        self.assertTrue(self.memory_systems.short.is_empty)

        self.memory_systems = MemorySystems(
            episodic=EpisodicMemory(capacity=8),
            episodic_agent=EpisodicMemory(capacity=3),
            semantic=SemanticMemory(capacity=4),
            short=ShortMemory(capacity=1),
        )

        to_return = self.memory_systems.return_as_a_dict_list()
        self.assertTrue(to_return["episodic"] == [])
        self.assertTrue(to_return["episodic_agent"] == [])
        self.assertTrue(to_return["semantic"] == [])
        self.assertTrue(to_return["short"] == [])

        self.memory_systems.episodic.add(["foo", "bar", "baz", 1])
        self.memory_systems.episodic_agent.add(["foo", "bar", "baz", 1])
        self.memory_systems.semantic.add(["foo", "bar", "baz", 1])
        self.memory_systems.short.add(["foo", "bar", "baz", 1])

        self.memory_systems.forget_all()
        self.assertTrue(self.memory_systems.episodic.is_empty)
        self.assertTrue(self.memory_systems.episodic_agent.is_empty)
        self.assertTrue(self.memory_systems.semantic.is_empty)
        self.assertTrue(self.memory_systems.short.is_empty)
