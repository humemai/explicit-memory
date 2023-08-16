"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""
import random
from typing import List

from ..memory import ShortMemory


def encode_observation(memory_systems: dict, policy: str, obs: dict) -> None:
    """Non RL policy of encoding an observation into a short-term memory.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "argmax" or "neural"
    obs: observation = {"human": <human>,
                        "object": <obj>,
                        "object_location": <obj_loc>}

    """
    if policy.lower() == "argmax":
        mem_short = ShortMemory.ob2short(obs)
    else:
        raise NotImplementedError

    memory_systems["short"].add(mem_short)


def manage_memory(memory_systems: dict, policy: str) -> None:
    """Non RL memory management policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "episodic", "semantic", "forget", "random", or "neural"

    """
    assert policy.lower() in [
        "episodic",
        "semantic",
        "forget",
        "random",
        "neural",
    ]
    if policy.lower() == "episodic":
        if memory_systems["episodic"].capacity != 0:
            if memory_systems["episodic"].is_full:
                memory_systems["episodic"].forget_oldest()
            mem_short = memory_systems["short"].get_oldest_memory()
            mem_epi = ShortMemory.short2epi(mem_short)
            memory_systems["episodic"].add(mem_epi)

    elif policy.lower() == "semantic":
        if memory_systems["semantic"].capacity != 0:
            if memory_systems["semantic"].is_full:
                memory_systems["semantic"].forget_weakest()
            mem_short = memory_systems["short"].get_oldest_memory()
            mem_sem = ShortMemory.short2sem(mem_short)
            memory_systems["semantic"].add(mem_sem)

    elif policy.lower() == "forget":
        pass

    elif policy.lower() == "random":
        action_number = random.choice([0, 1, 2])
        if action_number == 0:
            if memory_systems["episodic"].is_full:
                memory_systems["episodic"].forget_oldest()
            mem_short = memory_systems["short"].get_oldest_memory()
            mem_epi = ShortMemory.short2epi(mem_short)
            memory_systems["episodic"].add(mem_epi)

        elif action_number == 1:
            if memory_systems["semantic"].is_full:
                memory_systems["semantic"].forget_weakest()

            mem_short = memory_systems["short"].get_oldest_memory()
            mem_sem = ShortMemory.short2sem(mem_short)
            memory_systems["semantic"].add(mem_sem)

        else:
            pass

    elif policy.lower() == "neural":
        raise NotImplementedError

    else:
        raise ValueError

    memory_systems["short"].forget_oldest()


def answer_question(memory_systems: dict, policy: str, question: dict) -> str:
    """Non RL question answering policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
                     This policy does not need short-term memory
    policy: "episodic_semantic", "semantic_episodic", "episodic", "semantic",
            "random", or "neural",
    question: question = {"human": <human>, "object": <obj>}

    Returns
    -------
    pred: prediction

    """
    assert policy.lower() in [
        "episodic_semantic",
        "semantic_episodic",
        "episodic",
        "semantic",
        "random",
        "neural",
    ]
    pred_epi, _ = memory_systems["episodic"].answer_latest(question)
    pred_sem, _ = memory_systems["semantic"].answer_strongest(question)

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

    return pred


def explore(
    memory_systems: dict, policy: str, sub_graphs: List[int], previous_num: int = None
) -> str:
    """None RL exploration policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
                     This policy does not need short-term memory
    policy: "uniform_random", "one_by_one", or "neural"

    Returns
    -------
    num: sub-graph number

    """
    if policy.lower() == "uniform_random":
        num = random.choice(sub_graphs)
    elif policy.lower() == "one_by_one":
        if previous_num is None:
            num = sub_graphs[0]
        else:
            num = sub_graphs[(sub_graphs.index(previous_num) + 1) % len(sub_graphs)]
    elif policy.lower() == "neural":
        raise NotImplementedError
    else:
        raise ValueError

    return num
