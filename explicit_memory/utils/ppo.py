"""Utility functions for PPO."""

import logging
import os
import shutil
from collections import deque
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import torch
from IPython.display import clear_output
from tqdm.auto import tqdm

from .utils import is_running_notebook, list_duplicates_of, write_pickle, write_yaml

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def save_final_results(
    scores_all: dict,
    actor_losses: list,
    critic_losses: list,
    default_root_dir: str,
    self: object,
) -> None:
    """Save ppo train / val / test results."""
    results = {
        "train_score": scores_all["train"],
        "validation_score": [
            {
                "mean": round(np.mean(scores).item(), 2),
                "std": round(np.std(scores).item(), 2),
            }
            for scores in scores_all["val"]
        ],
        "test_score": {
            "mean": round(np.mean(scores_all["test"]).item(), 2),
            "std": round(np.std(scores_all["test"]).item(), 2),
        },
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
    }
    write_yaml(results, os.path.join(default_root_dir, "results.yaml"))
    write_pickle(self, os.path.join(default_root_dir, "agent.pkl"))


def select_action(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    state: dict,
    is_test: bool,
    states: list | None = None,
    actions: list | None = None,
    values: list | None = None,
    log_probs: list | None = None,
) -> tuple[int, list, list]:
    """Select an action from the given state.

    Args:
        actor: the actor model.
        state: the current state.
        is_test: if True, the agent is testing.
        critic: the critic model.
        states: a list of states (buffer).
        actions: a list of actions (buffer).
        values: a list of values (buffer).
        log_probs: a list of log probabilities (buffer).

    Returns:
        selected_action: the action to take.
        actor_probs: the actor probabilities.
        critic_value: the critic values.

    """
    action, dist = actor(np.array([state]))  # [state] is to add a dummy batch dimension

    if is_test:
        # Use argmax to select the action with the highest logit (probability) when
        # testing
        selected_action = dist.probs.argmax()
        # selected_action = dist.logits.argmax(dim=-1)
    else:
        # Sample an action during training
        selected_action = action

    value = critic(np.array([state]))

    if not is_test:
        states.append(deepcopy(state))
        actions.append(selected_action)
        values.append(value)
        log_probs.append(dist.log_prob(selected_action))

    selected_action = selected_action.detach().cpu().item()
    actor_probs = dist.probs.detach().cpu().tolist()[0]
    critic_value = value.detach().cpu().tolist()[0][0]

    return selected_action, actor_probs, critic_value


def compute_gae(
    next_value: list, rewards: list, masks: list, values: list, gamma: float, tau: float
) -> list:
    """Compute gae."""
    values = values + [next_value]
    gae = 0
    returns: deque[float] = deque()

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.appendleft(gae + values[step])
    return list(returns)


def yield_mini_batches(
    epoch: int,
    mini_batch_size: int,
    states: np.ndarray,
    actions: torch.Tensor,
    values: torch.Tensor,
    log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
):
    """Yield mini-batches.

    Args:
        epoch: the number of epochs.
        mini_batch_size: the mini-batch size.
        states: the states.
        actions: the actions.
        values: the values.
        log_probs: the log probabilities.
        returns: the returns.
        advantages: the advantages.

    Yields:
        states: the states.
        actions: the actions.
        values: the values.
        log_probs: the log probabilities.
        returns: the returns.
        advantages: the advantages.

    """
    batch_size = states.shape[0]
    for _ in range(epoch):
        for _ in range(batch_size // mini_batch_size):
            rand_ids = np.random.choice(batch_size, mini_batch_size)
            yield states[rand_ids], actions[rand_ids], values[rand_ids], log_probs[
                rand_ids
            ], returns[rand_ids], advantages[rand_ids]


def update_model(
    next_state: dict,
    states: list,
    actions: list,
    rewards: list,
    values: list,
    masks: list,
    log_probs: list,
    gamma: float,
    tau: float,
    epoch: int,
    batch_size: int,
    epsilon: float,
    entropy_weight: float,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    actor_optimizer: torch.optim.Adam,
    critic_optimizer: torch.optim.Adam,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update the model by gradient descent.

    This function uses the rollout buffer (states, actions, rewards, values, masks, and
    log_probs) to update the actor and critic models.

    Args:
        next_state: the next state.
        states: a list of states.
        actions: a list of actions.
        rewards: a list of rewards.
        values: a list of values.
        masks: a list of masks.
        log_probs: a list of log probabilities.
        gamma: the discount factor.
        tau: the gae parameter.
        epoch: the number of epochs.
        batch_size: the mini-batch size.
        epsilon: the clipping parameter.
        entropy_weight: the entropy weight.
        actor: the actor model.
        critic: the critic model.
        actor_optimizer: the actor optimizer.
        critic_optimizer: the critic optimizer.

    Returns:
        actor_loss: the actor loss.
        critic_loss: the critic loss.

    """
    assert len(states) == len(actions) == len(rewards) == len(values) == len(masks), (
        f"The length of states, actions, rewards, values, and masks must be the same, "
        f"but they are {len(states)}, {len(actions)}, {len(rewards)}, {len(values)}, "
        f"and {len(masks)}."
    )

    next_value = critic(np.array([next_state]))

    returns = compute_gae(next_value, rewards, masks, values, gamma, tau)

    # Batch them and detach them from the computation graph
    states = np.array(states)
    actions = torch.cat(actions)
    returns = torch.cat(returns).detach()
    values = torch.cat(values).detach()
    log_probs = torch.cat(log_probs).detach()

    advantages = returns - values

    actor_losses, critic_losses = [], []

    for state, action, old_value, old_log_prob, return_, adv in yield_mini_batches(
        epoch=epoch,
        mini_batch_size=batch_size,
        states=states,
        actions=actions,
        values=values,
        log_probs=log_probs,
        returns=returns,
        advantages=advantages,
    ):
        # calculate ratios
        _, dist = actor(state)

        log_prob = dist.log_prob(action)
        ratio = (log_prob - old_log_prob).exp()

        # actor_loss
        surr_loss = ratio * adv
        # print(f"surr_loss={surr_loss}")
        clipped_surr_loss = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * adv

        # entropy
        entropy = dist.entropy().mean()

        actor_loss = (
            -torch.min(surr_loss, clipped_surr_loss).mean() - entropy * entropy_weight
        )

        # critic_loss
        value = critic(state)
        # clipped_value = old_value + (value - old_value).clamp(-0.5, 0.5)
        critic_loss = (return_ - value).pow(2).mean()

        # train critic
        critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)

        # Clip gradients: gradients will be modified in-place
        # if gradient_max_norm is not None:
        #     clip_grad_norm_(critic.parameters(), max_norm=gradient_max_norm)
        critic_optimizer.step()

        # train actor
        actor_optimizer.zero_grad()
        actor_loss.backward()
        # if gradient_max_norm is not None:
        #     clip_grad_norm_(actor.parameters(), max_norm=gradient_max_norm)
        actor_optimizer.step()

        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())

    actor_loss = sum(actor_losses) / len(actor_losses)
    critic_loss = sum(critic_losses) / len(critic_losses)

    return actor_loss, critic_loss


def save_validation(
    scores: list,
    scores_all_val: list,
    default_root_dir: str,
    num_validation: int,
    val_filenames: list,
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    if_duplicate_take_first: bool = False,
) -> None:
    """Keep the best validation model.

    Args:
        scores: a list of validation scores for the current validation episode.
        scores_all_val: all val scores
        default_root_dir: the root directory where the results are saved.
        num_validation: the current validation episode.
        val_filenames: a list of filenames for the validation models.
        actor: the actor model.
        critic: the critic model.
        if_duplicate_take_first: if True, take the first duplicate model. This will take
            the higher training loss model. If False, take the last duplicate model.
            This will take the lower training loss model.
    """
    mean_score = round(np.mean(scores).item())
    # For actor and critic, create a directory for saving models
    dir_path = os.path.join(
        default_root_dir, f"episode={num_validation}_val-score={mean_score}"
    )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    actor_filename = os.path.join(dir_path, "actor.pt")
    critic_filename = os.path.join(dir_path, "critic.pt")
    torch.save(actor.state_dict(), actor_filename)
    torch.save(critic.state_dict(), critic_filename)
    filename = dir_path  # Use directory path as the identifier

    val_filenames.append(filename)
    scores_all_val.append(scores)

    scores_to_compare = []
    for filename in val_filenames:
        score = int(filename.split("val-score=")[-1].split(".pt")[0].split("/")[-1])
        scores_to_compare.append(score)

    indexes = list_duplicates_of(scores_to_compare, max(scores_to_compare))
    if if_duplicate_take_first:
        file_to_keep = val_filenames[indexes[0]]
    else:
        file_to_keep = val_filenames[indexes[-1]]

    for filename in val_filenames:
        if filename != file_to_keep:
            shutil.rmtree(filename)  # Remove the directory and all its contents
            val_filenames.remove(filename)


def save_states_actions_probs_values(
    states: list,
    actions: list,
    actor_probs: list,
    critic_values: list,
    default_root_dir: str,
    is_train_val_or_test: str,
    num_validation: int | None = None,
) -> None:
    """Save states and actions.

    Args:
        states: a list of states.
        actions: a list of actions.
        actor_probs: a list of actor probabilities.
        critic_values: a list of critic values.
        default_root_dir: the root directory where the results are saved.
        is_train_val_or_test: "val" or "test".
        num_validation: the current validation episode.

    """
    assert is_train_val_or_test in ["train", "val", "test"]
    if is_train_val_or_test == "train":
        filename = os.path.join(
            default_root_dir,
            "states_actions_actor_probs_critic_values_train.yaml",
        )
    elif is_train_val_or_test == "val":
        filename = os.path.join(
            default_root_dir,
            f"states_actions_actor_probs_critic_values_val_episode={num_validation}.yaml",
        )
    else:
        filename = os.path.join(
            default_root_dir, "states_actions_actor_probs_critic_values_test.yaml"
        )

    assert len(states) == len(actions) == len(actor_probs) == len(critic_values)
    to_save = [
        {"state": s, "action": a, "actor_probs": ap, "critic_value": cv}
        for s, a, ap, cv in zip(states, actions, actor_probs, critic_values)
    ]
    write_yaml(to_save, filename)


def plot_results(
    scores_all: dict,
    actor_losses: list[float],
    critic_losses: list[float],
    actor_probs_all: dict[str, list[list[float]]],
    critic_values_all: dict[str, list[float]],
    num_validation: int,
    number_of_actions: int,
    num_episodes: int,
    total_maximum_episode_rewards: int,
    default_root_dir: str,
    to_plot: str = "all",
    save_fig: bool = False,
) -> None:
    """Plot things for ppo training.

    Args:
        to_plot: what to plot:
            all: everything
            actor_loss: actor loss
            critic_loss: critic loss
            scores: train, val, and test scores
            actor_probs_train: actor probabilities for training
            actor_probs_val: actor probabilities for validation
            actor_probs_test: actor probabilities for test
            critic_values_train: critic values for training
            critic_values_val: critic values for validation
            critic_values_test: critic values for test

    """
    is_notebook = is_running_notebook()

    if is_notebook:
        clear_output(True)

    if to_plot == "all":
        plt.figure(figsize=(20, 20))

        plt.subplot(331)
        plt.title("Actor losses")
        plt.plot(actor_losses)
        plt.xlabel("update counts")

        plt.subplot(332)
        plt.title("Critic losses")
        plt.plot(critic_losses)
        plt.xlabel("update counts")

        plt.subplot(333)
        if scores_all["train"]:
            plt.title(
                f"episode {num_validation} out of {num_episodes} episodes. "
                f"training score: {scores_all['train'][-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(scores_all["train"], label="Training score")
            plt.xlabel("episode num")

        if scores_all["val"]:
            val_means = [round(np.mean(scores).item()) for scores in scores_all["val"]]
            plt.title(
                f"validation score: {val_means[-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(val_means, label="Validation score")
            plt.xlabel("episode num")

        if scores_all["test"]:
            plt.title(
                f"test score: {np.mean(scores_all['test'])} out of {total_maximum_episode_rewards}"
            )
            plt.plot(
                [round(np.mean(scores_all["test"]).item(), 2)]
                * len(scores_all["train"]),
                label="Test score",
            )
            plt.xlabel("episode num")
        plt.legend(loc="upper left")

        plt.subplot(334)
        plt.title("Actor probs, train")
        for action_number in range(number_of_actions):
            plt.plot(
                [actor_prob[action_number] for actor_prob in actor_probs_all["train"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

        plt.subplot(335)
        plt.title("Actor probs, val")
        for action_number in range(number_of_actions):
            plt.plot(
                [actor_prob[action_number] for actor_prob in actor_probs_all["val"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

        plt.subplot(336)
        plt.title("Actor probs, test")
        for action_number in range(number_of_actions):
            plt.plot(
                [actor_prob[action_number] for actor_prob in actor_probs_all["test"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

        plt.subplot(337)
        plt.title("Critic values, train")
        plt.plot(critic_values_all["train"])
        plt.xlabel("number of actions")

        plt.subplot(338)
        plt.title("Critic values, val")
        plt.plot(critic_values_all["val"])
        plt.xlabel("number of actions")

        plt.subplot(339)
        plt.title("Critic values, test")
        plt.plot(critic_values_all["test"])
        plt.xlabel("number of actions")

        plt.subplots_adjust(hspace=0.5)
        if save_fig:
            plt.savefig(os.path.join(default_root_dir, "plot.pdf"))

        if is_notebook:
            plt.show()
        else:
            console(**locals())
            plt.close("all")

    elif to_plot == "actor_loss":
        plt.figure()
        plt.title("Actor losses")
        plt.plot(actor_losses)
        plt.xlabel("update counts")

    elif to_plot == "critic_loss":
        plt.figure()
        plt.title("Critic losses")
        plt.plot(critic_losses)
        plt.xlabel("update counts")

    elif to_plot == "scores":
        plt.figure()

        if scores_all["train"]:
            plt.title(
                f"episode {num_validation} out of {num_episodes}. "
                f"training score: {scores_all['train'][-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(scores_all["train"], label="Training score")
            plt.xlabel("episode num")

        if scores_all["val"]:
            val_means = [round(np.mean(scores).item()) for scores in scores_all["val"]]
            plt.title(
                f"validation score: {val_means[-1]} out of {total_maximum_episode_rewards}"
            )
            plt.plot(val_means, label="Validation score")
            plt.xlabel("episode num")

        if scores_all["test"]:
            plt.title(
                f"test score: {np.mean(scores_all['test'])} out of {total_maximum_episode_rewards}"
            )
            plt.plot(
                [round(np.mean(scores_all["test"]).item(), 2)]
                * len(scores_all["train"]),
                label="Test score",
            )
            plt.xlabel("episode num")
        plt.legend(loc="upper left")

    elif to_plot == "actor_probs_train":
        plt.figure()
        plt.title("Actor probs, train")
        for action_number in range(number_of_actions):
            plt.plot(
                [actor_prob[action_number] for actor_prob in actor_probs_all["train"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

    elif to_plot == "actor_probs_val":
        plt.figure()
        plt.title("Actor probs, val")
        for action_number in range(number_of_actions):
            plt.plot(
                [actor_prob[action_number] for actor_prob in actor_probs_all["val"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

    elif to_plot == "actor_probs_test":
        plt.figure()
        plt.title("Actor probs, test")
        for action_number in range(number_of_actions):
            plt.plot(
                [actor_prob[action_number] for actor_prob in actor_probs_all["test"]],
                label=f"action {action_number}",
            )
        plt.legend(loc="upper left")
        plt.xlabel("number of actions")

    elif to_plot == "critic_values_train":
        plt.figure()
        plt.title("Critic values, train")
        plt.plot(critic_values_all["train"])
        plt.xlabel("number of actions")

    elif to_plot == "critic_values_val":
        plt.figure()
        plt.title("Critic values, val")
        plt.plot(critic_values_all["val"])
        plt.xlabel("number of actions")

    elif to_plot == "critic_values_test":
        plt.figure()
        plt.title("Critic values, test")
        plt.plot(critic_values_all["test"])
        plt.xlabel("number of actions")
    else:
        raise ValueError(f"to_plot={to_plot} is not valid.")


def console(
    scores_all: dict,
    actor_losses: list,
    critic_losses: list,
    num_validation: int,
    num_episodes: int,
    total_maximum_episode_rewards: int,
    **kwargs,
) -> None:
    """Print the dqn training to the console."""
    if scores_all["train"]:
        tqdm.write(
            f"episode {num_validation} out of {num_episodes}.\n"
            f"training score: "
            f"{scores_all['train'][-1]} out of {total_maximum_episode_rewards}"
        )

    if scores_all["val"]:
        val_means = [round(np.mean(scores).item()) for scores in scores_all["val"]]
        tqdm.write(
            f"validation score: {val_means[-1]} "
            f"out of {total_maximum_episode_rewards}"
        )

    if scores_all["test"]:
        tqdm.write(
            f"test score: {np.mean(scores_all['test'])} out of {total_maximum_episode_rewards}"
        )
    if actor_losses:
        tqdm.write(f"actor training loss: {actor_losses[-1]}")
    if critic_losses:
        tqdm.write(f"critic training loss: {critic_losses[-1]}\n")
    print()
