# Agent for RoomEnv-v1

This repo is to train an agent that interacts with the [RoomEnv-v1](https://github.com/tae898/room-env).
The agent is trained with DQN. See the paper ["A Machine with Short-Term, Episodic, and Semantic Memory Systems"](https://doi.org/10.1609/aaai.v37i1.25075) for more information.

## Prerequisites

1. A unix or unix-like x86 machine
1. python 3.8 or higher.
1. Running in a virtual environment (e.g., conda, virtualenv, etc.) is highly recommended so that you don't mess up with the system python.
1. Make sure you are in the same directory where this README is located, i.e., `RoomEnv-v0`
1. Install the `explicit_memory` python package, by running `pip install ..`
1. `pip install -r requirements.txt`

## Run training

```sh
python train.py
```

The hyperparameters can be configured in [`train.yaml`](./train.yaml). The training results with the
checkpoints will be saved at [`./training_results/`](./training_results/).

## Results

|                 Average loss, training.                 |           Average total rewards per episode, validation.           |              Average total rewards per episode, test.               |
| :-----------------------------------------------------: | :----------------------------------------------------------------: | :-----------------------------------------------------------------: |
| ![](./figures/des_size=l-capacity=32-train_loss-v1.png) | ![](./figures/des_size=l-capacity=32-val_total_reward_mean-v1.png) | ![](./figures/des_size=l-capacity=32-test_total_reward_mean-v1.png) |

|           Average total rewards, varying capacities, test.           |
| :------------------------------------------------------------------: |
| ![](./figures/des_size=l-capacity=all-test_total_reward_mean-v1.png) |

Also check out [`./models/`](./models) to see the saved training results.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
1. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
1. Run `make test && make style && make quality` in the root repo directory, to ensure code quality.
1. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
1. Push to the Branch (`git push origin feature/AmazingFeature`)
1. Open a Pull Request

## [Cite our paper](https://doi.org/10.1609/aaai.v37i1.25075)

```bibtex
@article{Kim_Cochez_Francois-Lavet_Neerincx_Vossen_2023,
  title={A Machine with Short-Term, Episodic, and Semantic Memory Systems}, volume={37},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/25075},
  DOI={10.1609/aaai.v37i1.25075},
  abstractNote={Inspired by the cognitive science theory of the explicit human memory systems, we have modeled an agent with short-term, episodic, and semantic memory systems, each of which is modeled with a knowledge graph. To evaluate this system and analyze the behavior of this agent, we designed and released our own reinforcement learning agent environment, “the Room”, where an agent has to learn how to encode, store, and retrieve memories to maximize its return by answering questions. We show that our deep Q-learning based agent successfully learns whether a short-term memory should be forgotten, or rather be stored in the episodic or semantic memory systems. Our experiments indicate that an agent with human-like memory systems can outperform an agent without this memory structure in the environment.},
  number={1},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Kim, Taewoon and Cochez, Michael and Francois-Lavet, Vincent and Neerincx, Mark and Vossen, Piek},
  year={2023},
  month={Jun.},
  pages={48-56}
}
```

## Cite our code

[![DOI](https://zenodo.org/badge/411241603.svg)](https://zenodo.org/badge/latestdoi/411241603)

## Authors

- [Taewoon Kim](https://taewoon.kim/)
- [Michael Cochez](https://www.cochez.nl/)
- [Vincent Francois-Lavet](http://vincent.francois-l.be/)
- [Mark Neerincx](https://ocw.tudelft.nl/teachers/m_a_neerincx/)
- [Piek Vossen](https://vossen.info/)

## License

[MIT](https://choosealicense.com/licenses/mit/)
