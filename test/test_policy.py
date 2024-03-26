import unittest

import numpy as np

from humemai.policy import *


class PolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.memory_systems = MemorySystems(
            episodic=EpisodicMemory(capacity=4),
            episodic_agent=EpisodicMemory(capacity=4),
            semantic=SemanticMemory(capacity=4),
            semantic_map=SemanticMemory(capacity=4),
            short=ShortMemory(capacity=1),
        )

    def test_encode_observation(self):
        obs = ["foo", "bar", "baz", 1]
        encode_observation(self.memory_systems, obs)
        self.assertEqual(self.memory_systems.short.get_oldest_memory(), obs)
        self.assertEqual(self.memory_systems.short.size, 1)
        self.assertTrue(self.memory_systems.episodic.is_empty)
        self.assertTrue(self.memory_systems.episodic_agent.is_empty)
        self.assertTrue(self.memory_systems.semantic.is_empty)

        with self.assertRaises(ValueError):
            encode_observation(self.memory_systems, obs)

    def test_explore(self):
        obs = ["foo", "bar", "baz", 1]
        encode_observation(self.memory_systems, obs)
        with self.assertRaises(AssertionError):
            explore(self.memory_systems, "random")

        self.memory_systems.short.forget_all()
        with self.assertRaises(ValueError):
            explore(self.memory_systems, "foo")

    def test_explore_random(self):
        obs = ["foo", "bar", "baz", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic", split_possessive=False)

        obs = ["agent", "bar", "baz", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic_agent", split_possessive=False)

        obs = ["foo", "bar", "baz", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic", split_possessive=False)

        action = explore(self.memory_systems, "random")
        self.assertTrue(action in ["north", "east", "south", "west", "stay"])

    def test_explore_avoid_walls(self):
        with self.assertRaises(IndexError):
            explore(self.memory_systems, "avoid_walls")

        obs = ["livingroom", "north", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic")
        obs = ["livingroom", "south", "wall", 2]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic")
        self.assertEqual(self.memory_systems.episodic.size, 2)

        obs = ["agent", "atlocation", "livingroom", 11]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic_agent")

        agent_current_location = self.memory_systems.episodic_agent.get_latest_memory()[
            2
        ]
        for _ in range(10):
            action = explore(self.memory_systems, "avoid_walls")
            self.assertIn(action, ["east", "west", "stay"])

        obs = ["livingroom", "east", "wall", 3]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic")

        obs = ["livingroom", "west", "wall", 5]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic_map")

        for _ in range(10):
            action = explore(self.memory_systems, "avoid_walls")
            self.assertEqual(action, "stay")

        # start over
        self.setUp()

        obs = ["livingroom", "north", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic_map")

        obs = ["livingroom", "east", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic")

        obs = ["livingroom", "west", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic")

        obs = ["livingroom", "south", "officeroom", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic_map")

        obs = ["officeroom", "north", "livingroom", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic_map")

        obs = ["officeroom", "west", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic_map")

        obs = ["officeroom", "south", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic")

        obs = ["officeroom", "east", "kitchen", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "semantic")

        obs = ["kitchen", "north", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic")

        obs = ["kitchen", "east", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic")

        obs = ["kitchen", "south", "wall", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic")

        obs = ["kitchen", "west", "officeroom", 1]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic")

        obs = ["agent", "atlocation", "livingroom", 11]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic_agent")

        for _ in range(10):
            action = explore(self.memory_systems, "avoid_walls")
            self.assertIn(action, ["south", "stay"])

        obs = ["agent", "atlocation", "officeroom", 12]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic_agent")

        for _ in range(10):
            action = explore(self.memory_systems, "avoid_walls")
            self.assertIn(action, ["north", "east", "stay"])

        obs = ["agent", "atlocation", "kitchen", 13]
        encode_observation(self.memory_systems, obs)
        manage_memory(self.memory_systems, "episodic_agent")

        for _ in range(10):
            action = explore(self.memory_systems, "avoid_walls")
            self.assertIn(action, ["west", "stay"])

    # def test_explore_neural(self):
    #     obs = ["foo", "bar", "baz", 1]
    #     encode_observation(self.memory_systems, obs)
    #     manage_memory(self.memory_systems, "episodic", split_possessive=False)
    #     with self.assertRaises(NotImplementedError):
    #         explore(self.memory_systems, "neural")

    def test_manage_memory(self):
        obs = ["foo", "bar", "baz", 1]
        encode_observation(self.memory_systems, obs)
        with self.assertRaises(AssertionError):
            manage_memory(self.memory_systems, "foo", split_possessive=False)

        self.memory_systems.short.forget_all()
        self.memory_systems.short.add(["foo", "bar", "baz", 1])
        with self.assertRaises(ValueError):
            manage_memory(self.memory_systems, "episodic_agent", split_possessive=False)

        self.memory_systems.short.forget_oldest()
        self.memory_systems.short.add(["agent", "atlocation", "officeroom", 5])
        manage_memory(self.memory_systems, "episodic_agent", split_possessive=True)
        self.assertEqual(
            self.memory_systems.episodic_agent.get_latest_memory(),
            ["agent", "atlocation", "officeroom", 5],
        )
        self.assertEqual(self.memory_systems.episodic.size, 0)
        self.assertEqual(self.memory_systems.episodic_agent.size, 1)
        self.assertEqual(self.memory_systems.semantic.size, 0)
        self.assertEqual(self.memory_systems.short.size, 0)

        self.memory_systems.short.add(
            ["tae's desk", "atlocation", "tae's officeroom", 3]
        )
        manage_memory(self.memory_systems, "episodic", split_possessive=True)
        self.assertEqual(self.memory_systems.episodic.size, 1)
        self.assertEqual(
            self.memory_systems.episodic.get_oldest_memory(),
            ["tae's desk", "atlocation", "tae's officeroom", 3],
        )
        self.assertEqual(self.memory_systems.episodic.size, 1)
        self.assertEqual(self.memory_systems.episodic_agent.size, 1)
        self.assertEqual(self.memory_systems.semantic.size, 0)
        self.assertEqual(self.memory_systems.short.size, 0)

        self.memory_systems.short.add(["livingroom", "north", "wall", 5])
        manage_memory(self.memory_systems, "forget", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 1)
        self.assertEqual(self.memory_systems.episodic_agent.size, 1)
        self.assertEqual(self.memory_systems.semantic.size, 0)
        self.assertEqual(self.memory_systems.short.size, 0)

        self.memory_systems.short.add(["tae's livingroom", "north", "tae's wall", 5])
        manage_memory(self.memory_systems, "semantic", split_possessive=True)
        self.assertEqual(
            self.memory_systems.semantic.entries, [["livingroom", "north", "wall", 1]]
        )
        self.assertEqual(self.memory_systems.episodic.size, 1)
        self.assertEqual(self.memory_systems.episodic_agent.size, 1)
        self.assertEqual(self.memory_systems.semantic.size, 1)
        self.assertEqual(self.memory_systems.short.size, 0)

        self.memory_systems.forget_all()
        self.memory_systems.short.add(["livingroom", "north", "wall", 5])
        manage_memory(self.memory_systems, "generalize", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 1)
        self.assertEqual(self.memory_systems.episodic_agent.size, 0)
        self.assertEqual(self.memory_systems.semantic.size, 0)
        self.assertEqual(self.memory_systems.short.size, 0)

        self.memory_systems.episodic.add(["livingroom", "north", "wall", 1])
        self.memory_systems.episodic.add(["livingroom", "north", "wall", 2])
        self.memory_systems.episodic.add(["livingroom", "north", "wall", 3])
        self.memory_systems.short.add(["foo", "bar", "baz", 10])
        self.assertTrue(self.memory_systems.episodic.is_full)
        manage_memory(self.memory_systems, "generalize", split_possessive=False)
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertEqual(
            self.memory_systems.episodic.get_oldest_memory(), ["foo", "bar", "baz", 10]
        )
        self.assertEqual(
            self.memory_systems.semantic.get_strongest_memory(),
            ["livingroom", "north", "wall", 4],
        )
        self.assertEqual(self.memory_systems.episodic.size, 1)
        self.assertEqual(self.memory_systems.episodic_agent.size, 0)
        self.assertEqual(self.memory_systems.semantic.size, 1)
        self.assertEqual(self.memory_systems.short.size, 0)

        self.memory_systems.short.add(["phone", "atlocation", "livingroom", 2])
        manage_memory(self.memory_systems, "generalize", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 2)
        self.assertEqual(
            self.memory_systems.episodic.entries,
            [["phone", "atlocation", "livingroom", 2], ["foo", "bar", "baz", 10]],
        )
        self.assertEqual(
            self.memory_systems.semantic.get_strongest_memory(),
            ["livingroom", "north", "wall", 4],
        )
        self.assertEqual(self.memory_systems.episodic.size, 2)
        self.assertEqual(self.memory_systems.episodic_agent.size, 0)
        self.assertEqual(self.memory_systems.semantic.size, 1)
        self.assertTrue(self.memory_systems.short.is_empty)

        self.memory_systems.episodic.add(["tae's toy", "atlocation", "room", 1])
        self.memory_systems.episodic.add(["toy", "tae's atlocation", "tae's room", 2])
        self.memory_systems.short.add(["foo", "bar", "baz", 15])
        self.assertTrue(self.memory_systems.episodic.is_full)
        manage_memory(self.memory_systems, "generalize", split_possessive=True)
        self.assertEqual(
            self.memory_systems.semantic.entries,
            [["toy", "atlocation", "room", 2], ["livingroom", "north", "wall", 4]],
        )
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 0)
        self.assertEqual(self.memory_systems.semantic.size, 2)
        self.assertTrue(self.memory_systems.short.is_empty)

        encode_observation(self.memory_systems, ["agent", "atlocation", "room1", 11])
        manage_memory(self.memory_systems, "episodic_agent", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 1)
        self.assertEqual(self.memory_systems.semantic.size, 2)
        self.assertTrue(self.memory_systems.short.is_empty)

        encode_observation(self.memory_systems, ["agent", "atlocation", "room2", 12])
        manage_memory(self.memory_systems, "episodic_agent", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 2)
        self.assertEqual(self.memory_systems.semantic.size, 2)
        self.assertTrue(self.memory_systems.short.is_empty)

        encode_observation(self.memory_systems, ["agent", "atlocation", "room2", 13])
        manage_memory(self.memory_systems, "episodic_agent", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 3)
        self.assertEqual(self.memory_systems.semantic.size, 2)
        self.assertTrue(self.memory_systems.short.is_empty)

        encode_observation(self.memory_systems, ["agent", "atlocation", "room3", 13])
        manage_memory(self.memory_systems, "episodic_agent", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 2)
        self.assertTrue(self.memory_systems.short.is_empty)

        encode_observation(self.memory_systems, ["agent", "atlocation", "room7", 7])
        manage_memory(self.memory_systems, "episodic_agent", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 2)
        self.assertTrue(self.memory_systems.short.is_empty)

        encode_observation(self.memory_systems, ["agent", "atlocation", "room3", 2])
        manage_memory(self.memory_systems, "episodic_agent", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 2)
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertEqual(
            self.memory_systems.episodic_agent.entries,
            [
                ["agent", "atlocation", "room3", 2],
                ["agent", "atlocation", "room2", 12],
                ["agent", "atlocation", "room2", 13],
                ["agent", "atlocation", "room3", 13],
            ],
        )

        encode_observation(self.memory_systems, ["foo", "bar", "baz", 0])

        with self.assertRaises(ValueError):
            manage_memory(self.memory_systems, "episodic_agent", split_possessive=True)

        self.memory_systems.short.forget_latest()
        encode_observation(self.memory_systems, ["livingroom", "south", "wall", 11])
        manage_memory(self.memory_systems, "generalize", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 4)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 2)
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertEqual(
            self.memory_systems.episodic.entries,
            [
                ["phone", "atlocation", "livingroom", 2],
                ["foo", "bar", "baz", 10],
                ["livingroom", "south", "wall", 11],
                ["foo", "bar", "baz", 15],
            ],
        )

        encode_observation(
            self.memory_systems, ["tae's livingroom", "south", "wall", 16]
        )
        manage_memory(self.memory_systems, "generalize", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 3)
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertEqual(
            self.memory_systems.episodic.entries,
            [
                ["phone", "atlocation", "livingroom", 2],
                ["livingroom", "south", "wall", 11],
                ["tae's livingroom", "south", "wall", 16],
            ],
        )
        self.assertTrue(
            (
                self.memory_systems.semantic.entries
                == [
                    ["foo", "bar", "baz", 2],
                    ["toy", "atlocation", "room", 2],
                    ["livingroom", "north", "wall", 4],
                ]
            )
            or (
                self.memory_systems.semantic.entries
                == [
                    ["toy", "atlocation", "room", 2],
                    ["foo", "bar", "baz", 2],
                    ["livingroom", "north", "wall", 4],
                ]
            ),
        )
        encode_observation(
            self.memory_systems, ["livingroom", "tae's south", "wall", 8]
        )
        manage_memory(self.memory_systems, "generalize", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 4)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 3)
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertEqual(
            self.memory_systems.episodic.entries,
            [
                ["phone", "atlocation", "livingroom", 2],
                ["livingroom", "tae's south", "wall", 8],
                ["livingroom", "south", "wall", 11],
                ["tae's livingroom", "south", "wall", 16],
            ],
        )
        encode_observation(
            self.memory_systems, ["livingroom", "tae's south", "tae's room", 4]
        )
        manage_memory(self.memory_systems, "generalize", split_possessive=False)
        self.assertEqual(self.memory_systems.episodic.size, 4)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 3)
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertEqual(
            self.memory_systems.episodic.entries,
            [
                ["livingroom", "tae's south", "tae's room", 4],
                ["livingroom", "tae's south", "wall", 8],
                ["livingroom", "south", "wall", 11],
                ["tae's livingroom", "south", "wall", 16],
            ],
        )
        encode_observation(self.memory_systems, ["foo", "bar", "qux", 0])
        manage_memory(self.memory_systems, "generalize", split_possessive=True)
        self.assertEqual(self.memory_systems.episodic.size, 2)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 4)
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertTrue(
            (
                self.memory_systems.semantic.entries
                == [
                    ["foo", "bar", "baz", 2],
                    ["toy", "atlocation", "room", 2],
                    ["livingroom", "south", "wall", 3],
                    ["livingroom", "north", "wall", 4],
                ]
            )
            or (
                self.memory_systems.semantic.entries
                == [
                    ["toy", "atlocation", "room", 2],
                    ["foo", "bar", "baz", 2],
                    ["livingroom", "south", "wall", 3],
                    ["livingroom", "north", "wall", 4],
                ]
            ),
        )
        self.assertEqual(
            self.memory_systems.episodic.entries,
            [
                ["foo", "bar", "qux", 0],
                ["livingroom", "tae's south", "tae's room", 4],
            ],
        )
        encode_observation(self.memory_systems, ["foo", "bar", "qux", 0])
        manage_memory(self.memory_systems, "generalize", split_possessive=True)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 4)
        self.assertTrue(self.memory_systems.short.is_empty)

        encode_observation(
            self.memory_systems, ["headset", "atlocation", "officeroom", 0]
        )
        manage_memory(self.memory_systems, "generalize", split_possessive=True)
        self.assertEqual(self.memory_systems.episodic.size, 4)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 4)
        self.assertTrue(self.memory_systems.short.is_empty)

        encode_observation(
            self.memory_systems, ["headset", "atlocation", "officeroom", 1]
        )
        manage_memory(self.memory_systems, "generalize", split_possessive=True)
        self.assertEqual(self.memory_systems.episodic.size, 3)
        self.assertEqual(self.memory_systems.episodic_agent.size, 4)
        self.assertEqual(self.memory_systems.semantic.size, 4)
        self.assertTrue(self.memory_systems.short.is_empty)
        self.assertEqual(
            self.memory_systems.episodic.entries,
            [
                ["headset", "atlocation", "officeroom", 0],
                ["headset", "atlocation", "officeroom", 1],
                ["livingroom", "tae's south", "tae's room", 4],
            ],
        )
        self.assertTrue(
            (
                self.memory_systems.semantic.entries
                == [
                    ["foo", "bar", "qux", 2],
                    ["toy", "atlocation", "room", 2],
                    ["livingroom", "south", "wall", 3],
                    ["livingroom", "north", "wall", 4],
                ]
            )
            or (
                self.memory_systems.semantic.entries
                == [
                    ["toy", "atlocation", "room", 2],
                    ["foo", "bar", "qux", 2],
                    ["livingroom", "south", "wall", 3],
                    ["livingroom", "north", "wall", 4],
                ]
            )
            or (
                self.memory_systems.semantic.entries
                == [
                    ["foo", "bar", "qux", 2],
                    ["foo", "bar", "baz", 2],
                    ["livingroom", "south", "wall", 3],
                    ["livingroom", "north", "wall", 4],
                ]
            )
            or (
                self.memory_systems.semantic.entries
                == [
                    ["foo", "bar", "baz", 2],
                    ["foo", "bar", "qux", 2],
                    ["livingroom", "south", "wall", 3],
                    ["livingroom", "north", "wall", 4],
                ]
            )
        )

        for _ in range(10):
            encode_observation(
                self.memory_systems, ["headset", "atlocation", "officeroom", 1]
            )
            manage_memory(self.memory_systems, "random", split_possessive=False)

        # with self.assertRaises(NotImplementedError):
        #     encode_observation(
        #         self.memory_systems, ["headset", "atlocation", "officeroom", 1]
        #     )
        #     manage_memory(self.memory_systems, "neural", split_possessive=False)

    def test_answer_question(self):
        self.memory_systems.short.add(["i", "am", "short", 42])
        with self.assertRaises(AssertionError):
            answer_question(
                self.memory_systems,
                policy="foo",
                question=["foo", "bar", "?", 42],
                split_possessive=False,
            )
        self.memory_systems.short.forget_all()

        # with self.assertRaises(NotImplementedError):
        #     answer_question(
        #         self.memory_systems,
        #         policy="neural",
        #         question=["foo", "bar", "?", 42],
        #         split_possessive=False,
        #     )
        answer = answer_question(
            self.memory_systems,
            policy="episodic_semantic",
            question=["foo", "bar", "?", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "none")

        self.memory_systems.episodic.add(["foo", "bar", "baz", 1])
        self.memory_systems.episodic.add(["foo", "bar", "qux", 2])
        self.memory_systems.episodic.add(["baz", "bar", "baz", 3])
        self.memory_systems.episodic.add(["qux", "bar", "baz", 2])

        self.memory_systems.semantic.add(["foo", "bar", "baz", 1])
        self.memory_systems.semantic.add(["foo", "bar", "qux", 2])
        self.memory_systems.semantic.add(["baz", "bar", "baz", 3])
        self.memory_systems.semantic.add(["qux", "bar", "baz", 2])

        answer = answer_question(
            self.memory_systems,
            policy="episodic_semantic",
            question=["foo", "bar", "?", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "qux")
        self.assertEqual(answer, answer.lower())
        answer = answer_question(
            self.memory_systems,
            policy="episodic",
            question=["foo", "bar", "?", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "qux")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="episodic_semantic",
            question=["?", "bar", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "baz")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="episodic",
            question=["?", "bar", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "baz")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="episodic_semantic",
            question=["foo", "?", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "bar")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="episodic",
            question=["foo", "?", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "bar")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="semantic_episodic",
            question=["foo", "bar", "?", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "qux")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="semantic",
            question=["foo", "bar", "?", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "qux")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="semantic_episodic",
            question=["?", "bar", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "baz")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="semantic",
            question=["?", "bar", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "baz")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="semantic_episodic",
            question=["foo", "?", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "bar")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="semantic",
            question=["foo", "?", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "bar")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="random",
            question=["foo", "bar", "?", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "qux")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="random",
            question=["?", "bar", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "baz")
        self.assertEqual(answer, answer.lower())

        answer = answer_question(
            self.memory_systems,
            policy="random",
            question=["foo", "?", "baz", 42],
            split_possessive=False,
        )
        self.assertEqual(answer, "bar")
        self.assertEqual(answer, answer.lower())
