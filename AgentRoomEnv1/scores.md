# Scores

This is common.

```python
des_size: str = "l",
total_episode_rewards: int = 128,
check_resources: bool = True,
capacity: dict = {"episodic": 16, "semantic": 16, "short": 1}
```

## Config 0

```python
seed: int = 42,
question_prob: int = 1.0,
allow_random_human: bool = False,
allow_random_question: bool = False,
```

| Experiment                                     | Pretrained (Mean ± Std) | Not Pretrained (Mean ± Std) | note                                                           |
| ---------------------------------------------- | ----------------------- | --------------------------- | -------------------------------------------------------------- |
| Episodic Only                                  | -                       | 44.56 ± 1.85                | -                                                              |
| Semantic Only                                  | -                       | 55.72 ± 1.61                | -                                                              |
| Random                                         | -                       | 37.64 ± 2.16                | -                                                              |
| DQN, No DDQN, No Dueling (w/ warm_start)       | 108.2 ± 5.01            | 91.28 ± 8.43                | -                                                              |
| AAAI Paper DQN                                 | 110.7                   | 89.3                        | -                                                              |
| AAAI Paper DDQN                                | 108.2                   | 90.2                        | -                                                              |
| AAAI Paper Dueling DQN                         | 102.7                   | 81.4                        | -                                                              |
| AAAI Paper Dueling DDQN                        | 109.3                   | 89.0                        | -                                                              |
| After Writing Tests                            | 103.84 ± 5.34           | 83.96 ± 4.21                | -                                                              |
| After Fixing Replay Buffer                     | 101.16 ± 6.56           | 89.35 ± 8.45                | -                                                              |
| After Refactoring                              | 105.6 ± 7.33            | 82.2 ± 19.48                | -                                                              |
| Training for 32 Episodes                       | 103.3 ± 3.2             | 93.5 ± 3.8                  | -                                                              |
| Training for 32 Episodes, DDQN + Dueling       | 101.5 ± 4.4             | 88.8 ± 7.9                  | -                                                              |
| Training for 32 Episodes, DDQN                 | 97.5 ± 4.2              | 80.9 ± 6.6                  | -                                                              |
| Training for 32 Episodes, Dueling              | 97.5 ± 5.4              | 83.0 ± 6.3                  | -                                                              |
| Training for 32 Episodes, Fix Last State Issue | 106.2 ± 7.3             | 92.2 ± 9.51                 | -                                                              |
| refactored with "question_interval-1"          | 94.96 ± 2.69            | 79.96 ± 7.24                | idk why but everytime I refactor, I don't get the same results |

## Config 1

```python
seed: int = 42,
question_prob: int = 1.0,
allow_random_human: bool = False,
allow_random_question: bool = True,
```

| Experiment    | Pretrained (Mean ± Std) | Not Pretrained (Mean ± Std) |
| ------------- | ----------------------- | --------------------------- |
| Episodic Only | -                       | -36.72 ± 4.14               |
| Semantic Only | -                       | 3.35 ± 1.67                 |
| Random        | -                       | -15.88 ± 4.42               |

## Config 2

```python
seed: int = 42,
question_prob: int = 1.0,
allow_random_human: bool = True,
allow_random_question: bool = False,
```

| Experiment    | Pretrained (Mean ± Std) | Not Pretrained (Mean ± Std) |
| ------------- | ----------------------- | --------------------------- |
| Episodic Only | -                       | 65.63 ± 2.47                |
| Semantic Only | -                       | 73.96 ± 6.19                |
| Random        | -                       | 53.6 ± 5.85                 |

## Config 3

```python
seed: int = 42,
allow_random_human: bool = True,
allow_random_question: bool = True,
default_root_dir: str = "./training_results/config3a/",
```

### Config 3.A

```python
question_prob: int = 1.0
```

| Experiment               | Pretrained (Mean ± Std) | Not Pretrained (Mean ± Std) |
| ------------------------ | ----------------------- | --------------------------- |
| Episodic Only            | -                       | -26.64 ± 3.13               |
| Semantic Only            | -                       | 30.95 ± 4.99                |
| Random                   | -                       | -5.2 ± 4.62                 |
| Training for 32 Episodes | 54.23 ± 2.11            | 14.68 ± 7.35                |

Alright obviously this is really hard to train.
But there is a hope! when pretraied, it works. So the initialization is important!

### Config 3.B

```python
question_prob: int = 0.1
```

| Experiment    | Pretrained (Mean ± Std) | Not Pretrained (Mean ± Std) |
| ------------- | ----------------------- | --------------------------- |
| Episodic Only | -                       | -3.2 ± 0.923                |
| Semantic Only | -                       | 2.96 ± 1.33                 |
| Random        | -                       | -0.339 ± 1.03               |

This doesn't work at all!

## Config 4

```python
"env_config": {
    "des_size": "l",
    "question_prob": 1.0,
    "allow_random_human": True,
    "allow_random_question": True,
    "question_interval": 16,
    "default_root_dir": "./training_results/question_interval-16/",
},
```

| Experiment               | Pretrained (Mean ± Std) | Not Pretrained (Mean ± Std) |
| ------------------------ | ----------------------- | --------------------------- |
| Episodic Only            | -                       | -32.04 ± 2.85               |
| Semantic Only            | -                       | 33.64 ± 4.61                |
| Random                   | -                       | -0.12 ± 6.29                |
| Training for 32 Episodes | 40.56 ± 3.75            | 10.96 ± 3.13                |
