import unittest

from humemai.memory import LongMemory, Memory, MemorySystems, ShortMemory


class TestMemory(unittest.TestCase):

    def setUp(self):
        self.memory = Memory(capacity=5)

    def test_initialization(self):
        self.assertEqual(len(self.memory), 0)
        self.assertEqual(self.memory.capacity, 5)
        self.assertTrue(self.memory.is_empty)
        self.assertFalse(self.memory.is_full)
        self.assertFalse(self.memory.is_frozen)

    def test_add_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertEqual(len(self.memory), 1)
        self.assertTrue(
            self.memory.has_memory(["Alice", "likes", "Bob", {"type": "episodic"}])
        )

    def test_capacity(self):
        self.assertEqual(self.memory.size, 0)
        self.assertFalse(self.memory.is_full)
        self.memory.increase_capacity(10)
        self.assertEqual(self.memory.capacity, 15)
        self.memory.decrease_capacity(5)
        self.assertEqual(self.memory.capacity, 10)

    def test_freeze_memory(self):
        self.assertFalse(self.memory.is_frozen)
        self.memory.freeze()
        self.assertTrue(self.memory.is_frozen)
        with self.assertRaises(ValueError):
            self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.memory.unfreeze()
        self.assertFalse(self.memory.is_frozen)
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertTrue(
            self.memory.has_memory(["Alice", "likes", "Bob", {"type": "episodic"}])
        )

    def test_forget_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertTrue(
            self.memory.has_memory(["Alice", "likes", "Bob", {"type": "episodic"}])
        )
        self.memory.forget(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertFalse(
            self.memory.has_memory(["Alice", "likes", "Bob", {"type": "episodic"}])
        )

    def test_query_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.memory.add(["Alice", "loves", "Charlie", {"type": "episodic"}])
        result = self.memory.query(["Alice", "?", "?", {"type": "episodic"}])
        self.assertEqual(len(result), 2)
        self.assertTrue(
            ["Alice", "likes", "Bob", {"type": "episodic"}] in result.to_list()
        )
        self.assertTrue(
            ["Alice", "loves", "Charlie", {"type": "episodic"}] in result.to_list()
        )

    def test_retrieve_random_memory(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        random_memory = self.memory.retrieve_random_memory()
        self.assertTrue(random_memory in self.memory.to_list())

    def test_memory_operations(self):
        self.memory.add(["Alice", "likes", "Bob", {"type": "episodic"}])
        self.assertEqual(self.memory.size, 1)
        self.memory.forget_all()
        self.assertEqual(self.memory.size, 0)

    def tearDown(self):
        del self.memory


class TestShortMemory(unittest.TestCase):

    def setUp(self):
        self.short_memory = ShortMemory(capacity=5)

    def test_add_memory(self):
        self.short_memory.add(["Alice", "likes", "Bob", {"current_time": 12345}])
        self.assertEqual(len(self.short_memory), 1)
        self.assertTrue(
            self.short_memory.has_memory(
                ["Alice", "likes", "Bob", {"current_time": 12345}]
            )
        )

    def test_ob2short_conversion(self):
        ob = ["Alice", "likes", "Bob", 12345]
        short_mem = ShortMemory.ob2short(ob)
        self.assertEqual(short_mem, ["Alice", "likes", "Bob", {"current_time": 12345}])

    def tearDown(self):
        del self.short_memory


class TestLongMemory(unittest.TestCase):

    def setUp(self):
        self.long_memory = LongMemory(capacity=10)

    def test_add_memory(self):
        self.long_memory.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        self.assertEqual(len(self.long_memory), 1)
        self.assertTrue(
            self.long_memory.has_memory(
                ["Alice", "likes", "Bob", {"timestamp": [12345]}]
            )
        )

    def test_pretrain_semantic(self):
        semantic_knowledge = [
            ["desk", "atlocation", "office"],
            ["chair", "atlocation", "office"],
        ]
        self.long_memory.pretrain_semantic(semantic_knowledge)
        self.assertEqual(len(self.long_memory), 2)

    def tearDown(self):
        del self.long_memory


class TestMemorySystems(unittest.TestCase):

    def setUp(self):
        short_memory = ShortMemory(capacity=5)
        long_memory = LongMemory(capacity=10)
        self.memory_systems = MemorySystems(short_memory, long_memory)

    def test_get_working_memory(self):
        self.memory_systems.short.add(
            ["Alice", "likes", "Bob", {"current_time": 12345}]
        )
        self.memory_systems.long.add(["Alice", "likes", "Bob", {"timestamp": [12345]}])
        working_memory = self.memory_systems.get_working_memory()
        self.assertEqual(len(working_memory), 1)

        self.memory_systems.long.add(
            ["Alice", "likes", "Alice", {"timestamp": [12345]}]
        )
        working_memory = self.memory_systems.get_working_memory()
        self.assertEqual(len(working_memory), 2)

    def tearDown(self):
        del self.memory_systems

    def test_get_working_memory_capacity_handling(self):

        # Fill up the short memory completely
        for i in range(5):
            self.memory_systems.short.add(
                ["Person" + str(i), "likes", "Object" + str(i), {"current_time": 12345}]
            )

        # Fill up the long memory almost completely (leave space for one more entry)
        for i in range(9):
            self.memory_systems.long.add(
                ["Person" + str(i), "likes", "Object" + str(i), {"timestamp": [12345]}]
            )

        # Add one more entry to long memory, exceeding the combined capacity
        self.memory_systems.long.add(
            ["Person9", "likes", "Object9", {"timestamp": [12345]}]
        )

        # Retrieve working memory and check total count
        working_memory = self.memory_systems.get_working_memory()
        self.assertEqual(
            len(working_memory), 10
        )  # 5 from short + 10 from long - 5 same
