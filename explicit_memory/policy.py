"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""
import random
from typing import List

from .memory import EpisodicMemory, MemorySystems, SemanticMemory, ShortMemory


def encode_observation(memory_systems: MemorySystems, obs: List[List]) -> None:
    """Non RL policy of encoding an observation into a short-term memory.

    At the moment, observation is the same as short-term memory. However, in the future
    we may want to encode the observation into a different format, e.g., when
    observatio is in the pixel space.

    Args
    ----
    MemorySystems
    obs: observation as a quadruple: [head, relation, tail, num]

    """
    mem_short = ShortMemory.ob2short(obs)
    memory_systems.short.add(mem_short)


def explore(memory_systems: MemorySystems, explore_policy: str) -> str:
    """Explore the room (sub-graph).

    Args
    ----
    memory_systems: MemorySystems
    explore_policy: "random", "avoid_walls", or "neural"

    Returns
    -------
    action: The exploration action to take.

    """
    assert memory_systems.short.is_empty, "Short-term memory should be empty."
    if explore_policy == "random":
        action = random.choice(["north", "east", "south", "west", "stay"])
    elif explore_policy == "avoid_walls":
        assert not memory_systems.episodic_agent.is_empty

        agent_current_location = memory_systems.episodic_agent.get_latest_memory()[2]

        memories_rooms = []
        MARKER = "^^^"  # to allow hashing for the set operation

        memories_rooms += [
            MARKER.join(entry[:-1])
            for entry in memory_systems.episodic.entries
            if entry[1] in ["north", "east", "south", "west"] and entry[2] != "wall"
        ]

        memories_rooms += [
            MARKER.join(entry[:-1])
            for entry in memory_systems.semantic.entries
            if entry[1] in ["north", "east", "south", "west"] and entry[2] != "wall"
        ]

        memories_rooms = [mem.split(MARKER) for mem in list(set(memories_rooms))]
        memories_rooms = [
            mem for mem in memories_rooms if mem[0] == agent_current_location
        ]

        if len(memories_rooms) == 0:
            action = random.choice(["north", "east", "south", "west", "stay"])
        else:
            action = random.choice(memories_rooms)[1]

    elif explore_policy == "neural":
        raise NotImplementedError
    else:
        raise ValueError("Unknown exploration policy.")

    assert action in ["north", "east", "south", "west", "stay"]

    return action


def manage_memory(
    memory_systems: MemorySystems,
    policy: str,
    split_possessive: bool = True,
) -> None:
    """Non RL memory management policy.

    Args
    ----
    MemorySystems
    policy: "episodic", "semantic", "generalize", "forget", "random", "neural", "agent"
    split_possessive: whether to split the possessive, i.e., 's, or not.

    """
    assert not memory_systems.short.is_empty
    assert policy.lower() in [
        "episodic",
        "semantic",
        "forget",
        "random",
        "generalize",
        "neural",
        "agent",
    ]
    if policy.lower() == "agent":
        mem_short = memory_systems.short.get_oldest_memory()
        if "agent" != mem_short[0]:
            raise ValueError("Agent is not in the memory.")
        assert memory_systems.episodic_agent.capacity > 0
        if memory_systems.episodic_agent.is_full:
            memory_systems.episodic_agent.forget_oldest()
        mem_epi = ShortMemory.short2epi(mem_short)
        memory_systems.episodic_agent.add(mem_epi)

    elif policy.lower() == "episodic":
        assert memory_systems.episodic.capacity != 0
        if memory_systems.episodic.is_full:
            memory_systems.episodic.forget_oldest()
        mem_short = memory_systems.short.get_oldest_memory()
        mem_epi = ShortMemory.short2epi(mem_short)
        memory_systems.episodic.add(mem_epi)

    elif policy.lower() == "semantic":
        assert memory_systems.semantic.capacity != 0
        if memory_systems.semantic.is_full:
            memory_systems.semantic.forget_weakest()
        mem_short = memory_systems.short.get_oldest_memory()
        mem_sem = ShortMemory.short2sem(mem_short, split_possessive=split_possessive)
        memory_systems.semantic.add(mem_sem)

    elif policy.lower() == "forget":
        pass

    elif policy.lower() == "generalize":
        assert (
            memory_systems.episodic.capacity != 0
            and memory_systems.semantic.capacity != 0
        )
        if memory_systems.episodic.is_full:
            mems_epi, mem_sem = memory_systems.episodic.find_similar_memories(
                split_possessive=split_possessive,
            )
            if mems_epi is None and mem_sem is None:
                memory_systems.episodic.forget_oldest()
            else:
                for mem_epi in mems_epi:
                    memory_systems.episodic.forget(mem_epi)

                if memory_systems.semantic.can_be_added(mem_sem)[0]:
                    memory_systems.semantic.add(mem_sem)
                else:
                    if memory_systems.semantic.is_full:
                        mem_sem_weakset = memory_systems.semantic.get_weakest_memory()
                        if mem_sem_weakset[-1] <= mem_sem[-1]:
                            memory_systems.semantic.forget_weakest()
                            memory_systems.semantic.add(mem_sem)
                        else:
                            pass

        mem_short = memory_systems.short.get_oldest_memory()
        mem_epi = ShortMemory.short2epi(mem_short)
        memory_systems.episodic.add(mem_epi)

    elif policy.lower() == "random":
        action_number = random.choice([0, 1, 2])

        if action_number == 0:
            assert memory_systems.episodic.capacity != 0
            if memory_systems.episodic.is_full:
                memory_systems.episodic.forget_oldest()
            mem_short = memory_systems.short.get_oldest_memory()
            mem_epi = ShortMemory.short2epi(mem_short)
            memory_systems.episodic.add(mem_epi)

        elif action_number == 1:
            assert memory_systems.semantic.capacity != 0
            if memory_systems.semantic.is_full:
                memory_systems.semantic.forget_weakest()
            mem_short = memory_systems.short.get_oldest_memory()
            mem_sem = ShortMemory.short2sem(mem_short)
            memory_systems.semantic.add(mem_sem)

        else:
            pass

    elif policy.lower() == "neural":
        raise NotImplementedError

    else:
        raise ValueError

    memory_systems.short.forget_oldest()


def answer_question(
    memory_systems: MemorySystems,
    policy: str,
    question: List[str],
    split_possessive: bool = True,
) -> str:
    """Non RL question answering policy.

    Args
    ----
    MemorySystems
    qa_policy: "episodic_semantic", "semantic_episodic", "episodic", "semantic",
            "random", or "neural",
    question: e.g., [laptop, atlocation, ?, current_time]
    split_possessive: whether to split the possessive, i.e., 's, or not.

    Returns
    -------
    pred: prediction

    """
    if (
        len(question) != 4
        and isinstance(question[-1], int)
        and not (all([isinstance(e, str) for e in question[:-1]]))
    ):
        raise ValueError("Question is not in the correct format.")

    assert memory_systems.short.is_empty
    assert policy.lower() in [
        "episodic_semantic",
        "semantic_episodic",
        "episodic",
        "semantic",
        "random",
        "neural",
    ]
    pred_epi, _ = memory_systems.episodic.answer_latest(question)
    pred_sem, _ = memory_systems.semantic.answer_strongest(question, split_possessive)

    if policy.lower() == "episodic_semantic":
        if pred_epi is None:
            pred = pred_sem
        else:
            pred = pred_epi
    elif policy.lower() == "semantic_episodic":
        if pred_sem is None:
            pred = pred_epi
        else:
            pred = pred_sem
    elif policy.lower() == "episodic":
        pred = pred_epi
    elif policy.lower() == "semantic":
        pred = pred_sem
    elif policy.lower() == "random":
        pred = random.choice([pred_epi, pred_sem])
    elif policy.lower() == "neural":
        raise NotImplementedError
    else:
        raise ValueError

    return str(pred).lower()
