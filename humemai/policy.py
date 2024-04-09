"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""

import random
from typing import Literal

import numpy as np
import torch

from .memory import EpisodicMemory, MemorySystems, SemanticMemory, ShortMemory
from .utils import argmax


def encode_observation(memory_systems: MemorySystems, obs: list[str | int]) -> None:
    """Non RL policy of encoding an observation into a short-term memory.

    At the moment, observation is the same as short-term memory. However, in the future
    we may want to encode the observation into a different format, e.g., when
    observation is in the pixel space.

    Args:
        MemorySystems
        obs: observation as a quadruple: [head, relation, tail, num]

    """
    mem_short = ShortMemory.ob2short(obs)
    memory_systems.short.add(mem_short)


def encode_all_observations(
    memory_systems: MemorySystems, obs_multiple: list[list[str | int]]
) -> None:
    """Non RL policy of encoding all observations into short-term memories.

    Args:
        MemorySystems
        obs_multiple: a list of observations

    """
    for obs in obs_multiple:
        mem_short = ShortMemory.ob2short(obs)
        memory_systems.short.add(mem_short)


def find_agent_current_location(memory_systems: MemorySystems) -> str:
    """Find the current location of the agent.

    If memory_systems has episodic_agent, then it is used to find the current location
    if not, it looks up the episodic. If fails, it looks up the semantic.
    If all fails, it returns None.

    Args:
        MemorySystems

    Returns:
        agent_current_location: str

    """
    if hasattr(memory_systems, "episodic_agent"):
        agent_current_location = memory_systems.episodic_agent.get_latest_memory()[2]
        return agent_current_location

    if hasattr(memory_systems, "episodic"):
        mems = [
            mem
            for mem in memory_systems.episodic.entries
            if mem[0] == "agent" and mem[1] == "atlocation"
        ]
        if len(mems) > 0:
            agent_current_location = mems[-1][2]
            return agent_current_location

    if hasattr(memory_systems, "semantic"):
        mems = [
            mem
            for mem in memory_systems.semantic.entries
            if mem[0] == "agent" and mem[1] == "atlocation"
        ]
        if len(mems) > 0:
            agent_current_location = mems[-1][2]
            return agent_current_location

    return None


def find_visited_locations(
    memory_systems: MemorySystems,
) -> dict[str, list[list[str, int]]]:
    """Find the locations that the agent has visited so far.

    Args:
        MemorySystems: MemorySystems

    Returns:
        visited_locations: a dictionary of a list of [location, time/strength] pairs.

    """
    visited_locations = {"episodic": [], "semantic": []}
    if hasattr(memory_systems, "episodic_agent"):
        pair = [
            [mem[2], mem[3]]
            for mem in memory_systems.episodic_agent.entries
            if mem[1] == "atlocation"
        ]
        visited_locations["episodic"].append(pair)

    for mem in memory_systems.episodic.entries:
        if mem[0] == "agent" and mem[1] == "atlocation":
            pair = [mem[2], mem[3]]
            visited_locations["episodic"].append(pair)

    # ascending order
    sorted(visited_locations["episodic"], key=lambda x: x[1])

    for mem in memory_systems.semantic.entries:
        if mem[0] == "agent" and mem[1] == "atlocation":
            pair = [mem[2], mem[3]]
            visited_locations["semantic"].append(pair)

    # ascending order
    sorted(visited_locations["semantic"], key=lambda x: x[1])

    return visited_locations


def explore(
    memory_systems: MemorySystems,
    explore_policy: str,
    explore_policy_model: torch.nn.Module | None = None,
) -> str:
    """Explore the room (sub-graph).

    Args:
        memory_systems: MemorySystems
        explore_policy: "random", "avoid_walls", or "neural"
        explore_policy_model: a neural network model for exploration policy.

    Returns:
        action: The exploration action to take.

    """
    assert memory_systems.short.is_empty, "Short-term memory should be empty."
    if explore_policy == "random":
        action = random.choice(["north", "east", "south", "west", "stay"])
    elif explore_policy == "avoid_walls":
        agent_current_location = find_agent_current_location(memory_systems)

        # no information about the agent's location
        if agent_current_location is None:
            action = random.choice(["north", "east", "south", "west", "stay"])

        # Get all the memories related to the current location
        mems = []

        # from the semantic map
        if hasattr(memory_systems, "semantic_map"):
            mems += [
                mem
                for mem in memory_systems.semantic_map.entries
                if mem[0] == agent_current_location
                and mem[1] in ["north", "east", "south", "west"]
            ]

        # from the semantic memory
        if hasattr(memory_systems, "semantic"):
            mems += [
                mem
                for mem in memory_systems.semantic.entries
                if mem[0] == agent_current_location
                and mem[1] in ["north", "east", "south", "west"]
            ]

        # from the episodic
        if hasattr(memory_systems, "episodic"):
            mems += [
                mem
                for mem in memory_systems.episodic.entries
                if mem[0] == agent_current_location
                and mem[1] in ["north", "east", "south", "west"]
            ]

        # we know the agent's current location but there is no memory about the map
        if len(mems) == 0:
            action = random.choice(["north", "east", "south", "west", "stay"])

        else:
            # we know the agent's current location and there is at least one memory
            # about the map and we want to avoid the walls

            to_take = []
            to_avoid = []

            for mem in mems:
                if mem[2].split("_")[0] == "room":
                    to_take.append(mem[1])
                elif mem[2] == "wall":
                    if mem[1] not in to_avoid:
                        to_avoid.append(mem[1])

            if len(to_take) > 0:
                action = random.choice(to_take)
            else:
                options = ["north", "east", "south", "west", "stay"]
                for e in to_avoid:
                    options.remove(e)

                action = random.choice(options)

    elif explore_policy == "new_room":
        # I think avoid_walls is not working well, since it's stochastic.
        # so imma try this.
        raise NotImplementedError

    elif explore_policy == "neural":
        state = memory_systems.return_as_a_dict_list()
        with torch.no_grad():
            q_values = (
                explore_policy_model(np.array([state]))
                .detach()
                .cpu()
                .numpy()
                .tolist()[0]
            )
        selected_action = argmax(q_values)
        assert selected_action in [0, 1, 2, 3, 4]

        action = ["north", "east", "south", "west", "stay"][selected_action]

    else:
        raise ValueError("Unknown exploration policy.")

    assert action in ["north", "east", "south", "west", "stay"]

    return action


def manage_memory(
    memory_systems: MemorySystems,
    policy: str,
    mm_policy_model: torch.nn.Module | None = None,
    mm_policy_model_type: Literal["actor", "q_function"] | None = None,
    split_possessive: bool = True,
) -> None:
    """Non RL memory management policy.

    Args:
        MemorySystems
        policy: "episodic", "semantic", "generalize", "forget", "random", "neural",
            "episodic_agent", or "semantic_map",
        mm_policy_model: a neural network model for memory management policy.
        mm_policy_model_type: depends wheter your RL algorithm used.
        split_possessive: whether to split the possessive, i.e., 's, or not.

    """
    if mm_policy_model is not None:
        assert mm_policy_model_type is not None

    def action_number_0():
        if hasattr(memory_systems, "episodic"):
            assert memory_systems.episodic.capacity != 0
            if memory_systems.episodic.is_full:
                memory_systems.episodic.forget_oldest()
            mem_short = memory_systems.short.get_oldest_memory()
            mem_epi = ShortMemory.short2epi(mem_short)
            memory_systems.episodic.add(mem_epi)

    def action_number_1():
        if hasattr(memory_systems, "semantic"):
            assert memory_systems.semantic.capacity != 0
            if memory_systems.semantic.is_full:
                memory_systems.semantic.forget_weakest()
            mem_short = memory_systems.short.get_oldest_memory()
            mem_sem = ShortMemory.short2sem(
                mem_short, split_possessive=split_possessive
            )
            memory_systems.semantic.add(mem_sem)

    assert not memory_systems.short.is_empty
    assert policy.lower() in [
        "episodic",
        "semantic",
        "forget",
        "random",
        "generalize",
        "neural",
        "episodic_agent",
        "semantic_map",
    ]
    if policy.lower() == "episodic_agent":
        if hasattr(memory_systems, "episodic_agent"):
            mem_short = memory_systems.short.get_oldest_memory()
            if "agent" != mem_short[0]:
                raise ValueError("This is not an agent location related memory!")
            assert memory_systems.episodic_agent.capacity > 0
            if memory_systems.episodic_agent.is_full:
                memory_systems.episodic_agent.forget_oldest()
            mem_epi = ShortMemory.short2epi(mem_short)
            memory_systems.episodic_agent.add(mem_epi)

    elif policy.lower() == "semantic_map":
        if hasattr(memory_systems, "semantic_map"):
            mem_short = memory_systems.short.get_oldest_memory()
            if mem_short[1] not in ["north", "east", "south", "west"]:
                raise ValueError("This is not a room-map-related memory.")
            assert memory_systems.semantic_map.capacity > 0
            if memory_systems.semantic_map.is_full:
                memory_systems.semantic_map.forget_weakest()
            mem_sem = ShortMemory.short2sem(
                mem_short, split_possessive=split_possessive
            )
            memory_systems.semantic_map.add(mem_sem)

    elif policy.lower() == "episodic":
        action_number_0()

    elif policy.lower() == "semantic":
        action_number_1()

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
            action_number_0()

        elif action_number == 1:
            action_number_1()

        else:
            pass

    elif policy.lower() == "neural":
        state = memory_systems.return_as_a_dict_list()
        with torch.no_grad():
            if mm_policy_model_type == "q_function":
                q_values = mm_policy_model(np.array([state])).detach().cpu().tolist()[0]
                selected_action = argmax(q_values)

            elif mm_policy_model_type == "actor":
                action, dist = mm_policy_model(np.array([state]))
                selected_action = dist.probs.argmax().detach().cpu().item()
            else:
                raise ValueError

        assert selected_action in [0, 1, 2]

        if selected_action == 0:
            action_number_0()

        elif selected_action == 1:
            action_number_1()

        else:
            pass

    else:
        raise ValueError

    memory_systems.short.forget_oldest()


def answer_question(
    memory_systems: MemorySystems,
    policy: str,
    question: list[str],
    split_possessive: bool = True,
) -> str:
    """Non RL question answering policy.

    Args:
        MemorySystems
        qa_policy: "episodic_semantic", "semantic_episodic", "episodic", "semantic",
                "random", or "neural",
        question: e.g., [laptop, atlocation, ?, current_time]
        split_possessive: whether to split the possessive, i.e., 's, or not.

    Returns:
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
    if hasattr(memory_systems, "episodic"):
        pred_epi, _ = memory_systems.episodic.answer_latest(question)
    else:
        pred_epi = None

    if hasattr(memory_systems, "semantic"):
        pred_sem, _ = memory_systems.semantic.answer_strongest(
            question, split_possessive
        )
    else:
        pred_sem = None

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
