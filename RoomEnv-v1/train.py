import logging

logger = logging.getLogger()
logger.disabled = True

from explicit_memory.utils import read_yaml
from agent import DQNAgent


hparams = read_yaml("train.yaml")

agent = DQNAgent(**hparams)
agent.train()
