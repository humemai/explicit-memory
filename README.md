# https://anonymous.4open.science/r/agent/

1. python>=3.10
2. First go to https://anonymous.4open.science/r/env/ and install the gymnasium environment
3. `pip install -e .`
4. Train
   1. `python train-mm.py` for the $\pi_{mm}$
   2. `python train-explore.py` for the $\pi_{explore}$
   3. `python train-explore-baseline.py` for the $\pi_{explore}$ (baseline with the history room size of 24)
5. See the trained results
   1. [./training_results](./training_results) for your trained results
   2. [./trained-agents](./trained-agents) for the already trained results