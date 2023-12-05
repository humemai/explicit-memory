# Scores

This is common.

```python
des_size: str = "l",
total_episode_rewards: int = 128,
check_resources: bool = True,
capacity: dict = {"episodic": 16, "semantic": 16, "short": 1}
```

## Config 1

```python
seed: int = 42,
question_prob: int = 1.0,
allow_random_human: bool = False,
allow_random_question: bool = False,
```

| Experiment                                     | Pretrained (Mean ± Std) | Not Pretrained (Mean ± Std) |
| ---------------------------------------------- | ----------------------- | --------------------------- |
| Episodic Only                                  | -                       | 44.56 ± 1.85                |
| Semantic Only                                  | -                       | 55.72 ± 1.61                |
| Random                                         | -                       | 37.64 ± 2.16                |
| DQN, No DDQN, No Dueling (w/ warm_start)       | 108.2 ± 5.01            | 91.28 ± 8.43                |
| AAAI Paper DQN                                 | 110.7                   | 89.3                        |
| AAAI Paper DDQN                                | 108.2                   | 90.2                        |
| AAAI Paper Dueling DQN                         | 102.7                   | 81.4                        |
| AAAI Paper Dueling DDQN                        | 109.3                   | 89.0                        |
| After Writing Tests                            | 103.84 ± 5.34           | 83.96 ± 4.21                |
| After Fixing Replay Buffer                     | 101.16 ± 6.56           | 89.35 ± 8.45                |
| After Refactoring                              | 105.6 ± 7.33            | 82.2 ± 19.48                |
| Training for 32 Episodes                       | 103.3 ± 3.2             | 93.5 ± 3.8                  |
| Training for 32 Episodes, DDQN + Dueling       | 101.5 ± 4.4             | 88.8 ± 7.9                  |
| Training for 32 Episodes, DDQN                 | 97.5 ± 4.2              | 80.9 ± 6.6                  |
| Training for 32 Episodes, Dueling              | 97.5 ± 5.4              | 83.0 ± 6.3                  |
| Training for 32 Episodes, Fix Last State Issue | 106.2 ± 7.3             | 92.2 ± 9.51                 |

## Config 2

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

## Config 3

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
question_prob: int = 1.0,
allow_random_human: bool = True,
allow_random_question: bool = True,
```

| Experiment    | Pretrained (Mean ± Std) | Not Pretrained (Mean ± Std) |
| ------------- | ----------------------- | --------------------------- |
| Episodic Only | -                       | -26.64 ± 3.13               |
| Semantic Only | -                       | 30.95 ± 4.99                |
| Random        | -                       | -5.2 ± 4.62                 |
