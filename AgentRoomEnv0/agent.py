import random
from itertools import count

from explicit_memory.memory import EpisodicMemory, SemanticMemory
from explicit_memory.utils import argmax, seed_everything


class HandcraftedAgent:
    """The handcrafted agent class.

    This can be more than one (colaborative) agent.

    """

    def __init__(
        self,
        seed: int,
        agent_type: str,
        forget_policy: str,
        answer_policy: str,
        episodic_capacity: int,
        semantic_capacity: int,
        env,
    ) -> None:
        """Initialize the agent class.

        Args:
            seed: random seed
            agent_type: episodic, semantic, episodic_semantic, or episodic_semantic_pretrain
            forget_policy: see the code
            answer_policy: see the code
            episodic_capacity: number of memories the episodic memory system can have.
            semantic_capacity: number of memories the semantic memory system can have.

        """
        self.seed = seed
        seed_everything(self.seed)
        self.agent_type = agent_type
        self.forget_policy = forget_policy
        self.answer_policy = answer_policy
        self.episodic_capacity = episodic_capacity
        self.semantic_capacity = semantic_capacity
        self.env = env

        self._check_attributes()
        self._init_memory_systems()

    def _check_attributes(self):
        """Run sanity check."""
        if self.agent_type == "episodic":
            assert self.forget_policy in ["oldest", "random"]
            assert self.answer_policy in ["latest", "random"]
            assert self.episodic_capacity > 0
            assert self.semantic_capacity == 0
        elif self.agent_type == "semantic":
            assert self.forget_policy in ["weakest", "random"]
            assert self.answer_policy in ["strongest", "random"]
            assert self.episodic_capacity == 0
            assert self.semantic_capacity > 0
        elif self.agent_type == "episodic_semantic":
            assert self.forget_policy in ["generalize", "random"]
            assert self.answer_policy in ["episem", "random"]
            assert self.episodic_capacity > 0
            assert self.semantic_capacity > 0
        elif self.agent_type == "episodic_semantic_pretrain":
            assert self.forget_policy in ["oldest", "random"]
            assert self.answer_policy in ["episem", "random"]
            assert self.episodic_capacity > 0
            assert self.semantic_capacity > 0
        else:
            raise ValueError

    def _init_memory_systems(self):
        """Initialize the memory systems."""
        if self.agent_type == "episodic":
            self.M_e = [
                EpisodicMemory(capacity=self.episodic_capacity, remove_duplicates=False)
                for _ in range(self.env.num_agents)
            ]
        elif self.agent_type == "semantic":
            self.M_s = [
                SemanticMemory(capacity=self.semantic_capacity)
                for _ in range(self.env.num_agents)
            ]
        elif self.agent_type == "episodic_semantic":
            self.M_e = [
                EpisodicMemory(capacity=self.episodic_capacity, remove_duplicates=False)
                for _ in range(self.env.num_agents)
            ]
            self.M_s = [
                SemanticMemory(capacity=self.semantic_capacity)
                for _ in range(self.env.num_agents)
            ]
        else:
            self.M_e = []
            self.M_s = []
            for _ in range(self.env.num_agents):
                me = EpisodicMemory(
                    capacity=self.episodic_capacity, remove_duplicates=False
                )
                ms = SemanticMemory(capacity=self.semantic_capacity)
                free_space = ms.pretrain_semantic(self.env.semantic_knowledge)
                me.increase_capacity(free_space)

                assert (
                    me.capacity + ms.capacity
                    == self.episodic_capacity + self.semantic_capacity
                )

                self.M_e.append(me)
                self.M_s.append(ms)

    def run(self):
        """RUN!"""
        if self.agent_type == "episodic":
            self.run_episodic()
        elif self.agent_type == "semantic":
            self.run_semantic()
        elif self.agent_type == "episodic_semantic":
            self.run_episodic_semantic()
        else:
            self.run_episodic_semantic_pretrain()

    def run_episodic(self):
        """Run an agent only with the episodic memory system."""
        self.rewards = 0
        (ob, question), info = self.env.reset()
        for i in range(self.env.num_agents):
            self.M_e[i].add(EpisodicMemory.ob2epi(ob[i]))

        for t in count():
            for i in range(self.env.num_agents):
                if self.M_e[i].is_full:
                    if self.forget_policy == "oldest":
                        self.M_e[i].forget_oldest()
                    elif self.forget_policy == "random":
                        self.M_e[i].forget_random()
                    else:
                        raise ValueError

            if self.answer_policy == "latest":
                preds = []
                timestamps = []
                for i in range(self.env.num_agents):
                    pred, timestamp = self.M_e[i].answer_latest(question)
                    if (pred is not None) and (timestamp is not None):
                        preds.append(pred)
                        timestamps.append(timestamp)

                if len(preds) > 0 and len(timestamps) > 0:
                    idx = argmax(timestamps)
                    pred = preds[idx]
                else:
                    pred = None

            elif self.answer_policy == "random":
                preds = []
                timestamps = []
                for i in range(self.env.num_agents):
                    pred, timestamp = self.M_e[i].answer_random(question)
                    if (pred is not None) and (timestamp is not None):
                        preds.append(pred)
                        timestamps.append(timestamp)

                if len(preds) > 0 and len(timestamps) > 0:
                    pred = random.choice(preds)
                else:
                    pred = None
            else:
                raise ValueError

            (ob, question), reward, done, truncated, info = self.env.step(pred)

            for i in range(self.env.num_agents):
                self.M_e[i].add(EpisodicMemory.ob2epi(ob[i]))

            self.rewards += reward

            if done:
                break

    def run_semantic(self):
        """Run an agent only with the semantic memory system."""
        self.rewards = 0
        (ob, question), info = self.env.reset()
        for i in range(self.env.num_agents):
            self.M_s[i].add(SemanticMemory.ob2sem(ob[i]))

        for t in count():
            for i in range(self.env.num_agents):
                if self.M_s[i].is_full:
                    if self.forget_policy == "weakest":
                        self.M_s[i].forget_weakest()
                    elif self.forget_policy == "random":
                        self.M_s[i].forget_random()
                    else:
                        raise ValueError

            if self.answer_policy == "strongest":
                preds = []
                num_gens = []
                for i in range(self.env.num_agents):
                    pred, num_gen = self.M_s[i].answer_strongest(question)
                    if (pred is not None) and (num_gen is not None):
                        preds.append(pred)
                        num_gens.append(num_gen)

                if len(preds) > 0 and len(num_gens) > 0:
                    idx = argmax(num_gens)
                    pred = preds[idx]
                else:
                    pred = None

            elif self.answer_policy == "random":
                preds = []
                num_gens = []
                for i in range(self.env.num_agents):
                    pred, num_gen = self.M_s[i].answer_random(question)
                    if (pred is not None) and (num_gen is not None):
                        preds.append(pred)
                        num_gens.append(num_gen)

                if len(preds) > 0 and len(num_gens) > 0:
                    pred = random.choice(preds)
                else:
                    pred = None
            else:
                raise ValueError

            (ob, question), reward, done, truncated, info = self.env.step(pred)

            for i in range(self.env.num_agents):
                self.M_s[i].add(SemanticMemory.ob2sem(ob[i]))

            self.rewards += reward

            if done:
                break

    def run_episodic_semantic(self):
        """Run an agent both with the episodic and semantic memory system."""
        self.rewards = 0
        env_started = False
        pred = None

        for t in count():
            if env_started:
                (ob, question), reward, done, truncated, info = self.env.step(pred)
                self.rewards += reward
                if done:
                    break

            else:
                (ob, question), info = self.env.reset()
                env_started = True

            for i in range(self.env.num_agents):
                if self.M_e[i].is_full:
                    if self.forget_policy == "generalize":
                        mems_epi, mem_sem = self.M_e[i].find_similar_memories()
                        if mems_epi is None and mem_sem is None:
                            self.M_e[i].forget_oldest()
                        else:
                            for mem_epi in mems_epi:
                                self.M_e[i].forget(mem_epi)

                            if self.M_s[i].can_be_added(mem_sem)[0]:
                                self.M_s[i].add(mem_sem)

                            else:
                                if self.M_s[i].is_full:
                                    mem_sem_weakset = self.M_s[i].get_weakest_memory()
                                    if mem_sem_weakset[-1] <= mem_sem[-1]:
                                        self.M_s[i].forget_weakest()
                                        self.M_s[i].add(mem_sem)
                                    else:
                                        pass

                    elif self.forget_policy == "random":
                        self.M_e[i].forget_random()

                    else:
                        raise ValueError

                if self.forget_policy == "generalize":
                    self.M_e[i].add(EpisodicMemory.ob2epi(ob[i]))
                elif self.forget_policy == "random":
                    if random.random() < 0.5:
                        self.M_e[i].add(EpisodicMemory.ob2epi(ob[i]))
                    else:
                        if self.M_s[i].is_full:
                            assert self.forget_policy == "random"
                            self.M_s[i].forget_weakest()
                        self.M_s[i].add(SemanticMemory.ob2sem(ob[i]))

            if self.answer_policy == "episem":
                preds = []
                timestamps = []
                preds_ = []
                num_gens = []
                for i in range(self.env.num_agents):
                    pred, timestamp = self.M_e[i].answer_latest(question)
                    if pred is not None and timestamp is not None:
                        preds.append(pred)
                        timestamps.append(timestamp)
                    else:
                        pred, num_gen = self.M_s[i].answer_strongest(question)
                        if pred is not None and num_gen is not None:
                            preds_.append(pred)
                            num_gens.append(num_gen)

                if len(preds) > 0 and len(timestamps) > 0:
                    idx = argmax(timestamps)
                    pred = preds[idx]

                else:
                    if len(preds_) > 0 and len(num_gens) > 0:
                        idx = argmax(num_gens)
                        pred = preds_[idx]
                    else:
                        pred = None

            elif self.answer_policy == "random":
                preds = []
                for i in range(self.env.num_agents):
                    pred, _ = self.M_e[i].answer_random(question)
                    preds.append(pred)
                    pred, _ = self.M_s[i].answer_random(question)
                    preds.append(pred)
                pred = random.choice(preds)
            else:
                raise ValueError

    def run_episodic_semantic_pretrain(self):
        """Run an agent both with the episodic and pretrained semantic memory system."""
        self.rewards = 0
        env_started = False
        pred = None

        for t in count():
            if env_started:
                (ob, question), reward, done, truncated, info = self.env.step(pred)
                self.rewards += reward
                if done:
                    break
            else:
                (ob, question), info = self.env.reset()
                env_started = True

            for i in range(self.env.num_agents):
                if self.M_e[i].is_full:
                    if self.forget_policy == "oldest":
                        self.M_e[i].forget_oldest()

                    elif self.forget_policy == "random":
                        self.M_e[i].forget_random()

                    else:
                        raise ValueError

                self.M_e[i].add(EpisodicMemory.ob2epi(ob[i]))

            if self.answer_policy == "episem":
                preds = []
                timestamps = []
                preds_ = []
                num_gens = []
                for i in range(self.env.num_agents):
                    pred, timestamp = self.M_e[i].answer_latest(question)
                    if pred is not None and timestamp is not None:
                        preds.append(pred)
                        timestamps.append(timestamp)
                    else:
                        pred, num_gen = self.M_s[i].answer_strongest(question)
                        if pred is not None and num_gen is not None:
                            preds_.append(pred)
                            num_gens.append(num_gen)

                if len(preds) > 0 and len(timestamps) > 0:
                    idx = argmax(timestamps)
                    pred = preds[idx]

                else:
                    if len(preds_) > 0 and len(num_gens) > 0:
                        idx = argmax(num_gens)
                        pred = preds_[idx]
                    else:
                        pred = None

            elif self.answer_policy == "random":
                preds = []
                for i in range(self.env.num_agents):
                    pred, _ = self.M_e[i].answer_random(question)
                    preds.append(pred)
                    pred, _ = self.M_s[i].answer_random(question)
                    preds.append(pred)
                pred = random.choice(preds)
            else:
                raise ValueError
