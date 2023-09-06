# import random
# import unittest

# import gymnasium as gym
# import room_env
# from model import LSTM
# from train import ReplayBuffer, RLAgent


# class RLAgentTest(unittest.TestCase):
#     def test_all(self) -> None:
#         for pretrain_semantic in [True, False]:
#             env = gym.make("room_env:RoomEnv-v1")
#             replay_buffer = ReplayBuffer(16)
#             agent = RLAgent(
#                 env=env,
#                 replay_buffer=replay_buffer,
#                 capacity={"episodic": 1, "semantic": 1, "short": 1},
#                 pretrain_semantic=pretrain_semantic,
#                 policies={
#                     "memory_management": "Rl",
#                     "question_answer": "episodic_semantic",
#                     "encoding": "argmax",
#                 },
#             )

#             config = {
#                 "hidden_size": 64,
#                 "num_layers": 2,
#                 "n_actions": 3,
#                 "embedding_dim": 32,
#                 "capacity": {
#                     "episodic": 16,
#                     "semantic": 16,
#                     "short": 1,
#                 },
#                 "entities": {
#                     "humans": env.des.humans,
#                     "objects": env.des.objects,
#                     "object_locations": env.des.object_locations,
#                 },
#                 "include_human": "sum",
#             }
#             lstm = LSTM(**config)

#             for _ in range(10):
#                 reward, done = agent.play_step(net=lstm, epsilon=random.random())
