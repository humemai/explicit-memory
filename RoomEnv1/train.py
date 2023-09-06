import logging

logger = logging.getLogger()
logger.disabled = True

from agent import DQNAgent

from explicit_memory.utils import read_yaml

hparams = read_yaml("train.yaml")

agent = DQNAgent(**hparams)
agent.train()
