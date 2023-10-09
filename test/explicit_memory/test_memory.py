import random
import unittest

import numpy as np

from explicit_memory.memory import *


class MemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = Memory(memory_type="episodic", capacity=8)

    def test_init(self):
        with self.assertRaises(AssertionError):
            foo = Memory(memory_type="foo", capacity=0)

    def test_add(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.assertEqual(self.memory.size, 1)

        self.memory.freeze()
        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 1])

        timestamps = [mem[-1] for mem in self.memory.entries]
        self.assertEqual(sorted(timestamps), timestamps)

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
        mems_found[0][0] = "qux"
        self.assertEqual(self.memory.entries[0], ["foo", "bar", "baz", 1])

    def test_is_valid_query(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        self.assertFalse(self.memory.is_valid_query(["foo", "bar", "baz"]))
        self.assertFalse(self.memory.is_valid_query(["foo", "bar", "baz", "qux"]))
        self.assertFalse(self.memory.is_valid_query(["foo", "?", "?"]))
        self.assertFalse(self.memory.is_valid_query(["foo", "?", 42]))

        self.assertTrue(self.memory.is_valid_query(["foo", "bar", "?"]))
        self.assertTrue(self.memory.is_valid_query(["foo", "?", "baz"]))
        self.assertTrue(self.memory.is_valid_query(["?", "bar", "baz"]))

    def test_answer_random(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])

        self.assertEqual(self.memory.answer_random(["foo", "bar", "?"])[0], "baz")

    def test_answer_with_smallest_num(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_with_smallest_num(["foo", "bar", "?"])
        self.assertEqual((pred, num), ("baz", 1))

        pred, num = self.memory.answer_with_smallest_num(["foo", "baz", "?"])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_with_smallest_num(["?", "bar", "baz"])
        self.assertEqual((pred, num), ("foo", 1))

        pred, num = self.memory.answer_with_smallest_num(["?", "foo", "bar"])
        self.assertEqual((pred, num), ("baz", 5))

    def test_answer_with_largest_num(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_with_largest_num(["foo", "bar", "?"])
        self.assertEqual((pred, num), ("baz", 3))

        pred, num = self.memory.answer_with_largest_num(["foo", "baz", "?"])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_with_largest_num(["?", "bar", "baz"])
        self.assertEqual((pred, num), ("foo", 3))

        pred, num = self.memory.answer_with_largest_num(["?", "foo", "bar"])
        self.assertEqual((pred, num), ("baz", 5))


class EpisodicMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory = EpisodicMemory(capacity=8, remove_duplicates=False)

    def test_can_be_added(self):
        for idx in range(8):
            mem_random = [
                str(random.random()),
                str(random.random()),
                str(random.random()),
                str(random.random()),
            ]
            self.assertTrue(self.memory.can_be_added(mem_random))
            self.memory.add(mem_random)

        mem_random = [
            str(random.random()),
            str(random.random()),
            str(random.random()),
            str(random.random()),
        ]
        self.assertFalse(self.memory.can_be_added(mem_random)[0])

    def test_add(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.assertEqual(self.memory.size, 1)

        self.memory.freeze()
        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 1])

    def test_get_oldest_memory(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        oldest = self.memory.get_oldest_memory()
        self.assertEqual(oldest, ["foo", "bar", "baz", 1])
        oldest[-1] = 1234
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
        latest[-1] = 1234
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

        pred, num = self.memory.answer_oldest(["foo", "bar", "?"])
        self.assertEqual((pred, num), ("baz", 1))

        pred, num = self.memory.answer_oldest(["foo", "baz", "?"])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_oldest(["?", "bar", "baz"])
        self.assertEqual((pred, num), ("foo", 1))

        pred, num = self.memory.answer_oldest(["?", "foo", "bar"])
        self.assertEqual((pred, num), ("baz", 5))

    def test_answer_latest(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_latest(["foo", "bar", "?"])
        self.assertEqual((pred, num), ("baz", 3))

        pred, num = self.memory.answer_latest(["foo", "baz", "?"])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_latest(["?", "bar", "baz"])
        self.assertEqual((pred, num), ("foo", 3))

        pred, num = self.memory.answer_latest(["?", "foo", "bar"])
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

    def test_can_be_added(self):
        for idx in range(8):
            mem_random = [
                str(random.random()),
                str(random.random()),
                str(random.random()),
                str(random.random()),
            ]
            self.assertTrue(self.memory.can_be_added(mem_random))
            self.memory.add(mem_random)

        mem_random = [
            str(random.random()),
            str(random.random()),
            str(random.random()),
            str(random.random()),
        ]
        self.assertFalse(self.memory.can_be_added(mem_random)[0])

    def test_add(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.assertEqual(self.memory.size, 1)

        self.memory.freeze()
        with self.assertRaises(ValueError):
            self.memory.add(["foo", "bar", "baz", 1])

    def test_get_oldest_memory(self):
        self.memory.add(["foo", "bar", "baz", 4])
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 3])

        oldest = self.memory.get_oldest_memory()
        self.assertEqual(oldest, ["foo", "bar", "baz", 4])
        oldest[-1] = 1234
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
        latest[-1] = 1234
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

        pred, num = self.memory.answer_oldest(["foo", "bar", "?"])
        self.assertEqual((pred, num), ("baz", 3))

        pred, num = self.memory.answer_oldest(["foo", "baz", "?"])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_oldest(["?", "bar", "baz"])
        self.assertEqual((pred, num), ("foo", 3))

        pred, num = self.memory.answer_oldest(["?", "foo", "bar"])
        self.assertEqual((pred, num), ("baz", 5))

    def test_answer_latest(self):
        self.memory.add(["foo", "bar", "baz", 1])
        self.memory.add(["foo", "bar", "baz", 2])
        self.memory.add(["foo", "bar", "baz", 3])
        self.memory.add(["baz", "foo", "bar", 5])

        pred, num = self.memory.answer_latest(["foo", "bar", "?"])
        self.assertEqual((pred, num), ("baz", 3))

        pred, num = self.memory.answer_latest(["foo", "baz", "?"])
        self.assertEqual((pred, num), (None, None))

        pred, num = self.memory.answer_latest(["?", "bar", "baz"])
        self.assertEqual((pred, num), ("foo", 3))

        pred, num = self.memory.answer_latest(["?", "foo", "bar"])
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

    def test_can_be_added(self) -> None:
        for idx in range(8):
            mem_random = [
                str(random.random()),
                str(random.random()),
                str(random.random()),
                random.randint(0, 100),
            ]
            self.assertTrue(self.memory.can_be_added(mem_random))
            self.memory.add(mem_random)

        mem_random = [
            str(random.random()),
            str(random.random()),
            str(random.random()),
            random.randint(0, 100),
        ]
        self.assertFalse(self.memory.can_be_added(mem_random)[0])
