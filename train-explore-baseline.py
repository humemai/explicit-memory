import matplotlib

matplotlib.use("Agg")
import argparse
import logging

logger = logging.getLogger()
logger.disabled = True

from agent.dqn import DQNLSTMBaselineAgent

from explicit_memory.utils import read_yaml

parser = argparse.ArgumentParser(description="train")
parser.add_argument(
    "-c",
    "--config",
    default="./train-explore-baseline.yaml",
    type=str,
    help="config file path (default: ./train-explore-baseline.yaml)",
)
args = parser.parse_args()
hparams = read_yaml(args.config)
print("Arguments:")
for k, v in hparams.items():
    print(f"  {k:>21} : {v}")


agent = DQNLSTMBaselineAgent(**hparams)
agent.train()