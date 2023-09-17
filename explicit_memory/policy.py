"""Handcrafted / trained policies.

The trained neural network policies are not implemented yet.
"""
import random
from typing import List

from .memory import EpisodicMemory, SemanticMemory, ShortMemory


def encode_observation(memory_systems: dict, obs: List[List]) -> None:
    """Non RL policy of encoding an observation into a short-term memory.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    obs: observation as a quadruple: [head, relation, tail, num]

    """
    mem_short = ShortMemory.ob2short(obs)
    memory_systems["short"].add(mem_short)


def manage_memory(
    memory_systems: dict,
    policy: str,
    dont_generalize_agent: bool = True,
    split_possessive: bool = True,
) -> None:
    """Non RL memory management policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
    policy: "episodic", "semantic", "generalize", "forget", "random", or "neural"
    dont_generalize_agent: if True, the agent-related memories are not generalized,
        i.e., they are not put into the semantic memory system.
    split_possessive: whether to split the possessive, i.e., 's, or not.

    """
    assert policy.lower() in [
        "episodic",
        "semantic",
        "forget",
        "random",
        "generalize",
        "neural",
    ]
    if policy.lower() == "episodic":
        assert memory_systems["episodic"].capacity != 0
        if memory_systems["episodic"].is_full:
            memory_systems["episodic"].forget_oldest()
        mem_short = memory_systems["short"].get_oldest_memory()
        mem_epi = ShortMemory.short2epi(mem_short)
        memory_systems["episodic"].add(mem_epi)

    elif policy.lower() == "semantic":
        assert memory_systems["semantic"].capacity != 0
        if memory_systems["semantic"].is_full:
            memory_systems["semantic"].forget_weakest()
        mem_short = memory_systems["short"].get_oldest_memory()
        mem_sem = ShortMemory.short2sem(mem_short, split_possessive=split_possessive)

        if dont_generalize_agent and mem_sem[0] == "agent":
            if (
                mem_sem[0] != "agent"
                and mem_sem[1] != "agent"
                and mem_sem[2] != "agent"
            ):
                memory_systems["semantic"].add(mem_sem)
        else:
            memory_systems["semantic"].add(mem_sem)

    elif policy.lower() == "forget":
        pass

    elif policy.lower() == "generalize":
        assert (
            memory_systems["episodic"].capacity != 0
            and memory_systems["semantic"].capacity != 0
        )
        if memory_systems["episodic"].is_full:
            mems_epi, mem_sem = memory_systems["episodic"].find_similar_memories(
                split_possessive=split_possessive,
                dont_generalize_agent=dont_generalize_agent,
            )
            if mems_epi is None and mem_sem is None:
                memory_systems["episodic"].forget_oldest()
            else:
                for mem_epi in mems_epi:
                    memory_systems["episodic"].forget(mem_epi)

                mem_sem_same = memory_systems["semantic"].find_same_memory(mem_sem)

                if mem_sem_same is not None:
                    memory_systems["semantic"].add(mem_sem)
                else:
                    if memory_systems["semantic"].is_full:
                        mem_sem_weakset = memory_systems[
                            "semantic"
                        ].get_weakest_memory()
                        if mem_sem_weakset[-1] <= mem_sem[-1]:
                            memory_systems["semantic"].forget_weakest()
                            memory_systems["semantic"].add(mem_sem)
                        else:
                            pass
                    else:
                        memory_systems["semantic"].add(mem_sem)

        mem_short = memory_systems["short"].get_oldest_memory()
        mem_epi = ShortMemory.short2epi(mem_short)
        memory_systems["episodic"].add(mem_epi)

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


def answer_question(
    memory_systems: dict, policy: str, question: dict, split_possessive: bool = True
) -> str:
    """Non RL question answering policy.

    Args
    ----
    memory_systems: {"episodic": EpisodicMemory, "semantic": SemanticMemory,
                     "short": ShortMemory}
                     This policy does not need short-term memory
    policy: "episodic_semantic", "semantic_episodic", "episodic", "semantic",
            "random", or "neural",
    question: question = {"human": <human>, "object": <obj>}
    split_possessive: whether to split the possessive, i.e., 's, or not.

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
    pred_sem, _ = memory_systems["semantic"].answer_strongest(
        question, split_possessive
    )

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
