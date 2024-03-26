"""Utility functions."""

import csv
import json
import logging
import os
import pickle
import random
import shutil
from glob import glob
from typing import Union

import numpy as np
import torch
import yaml

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def remove_timestamp(entry: list[str]) -> list:
    """Remove the timestamp from a given observation/episodic memory.

    Args:
        entry: An observation / episodic memory in a quadruple format
            (i.e., (head, relation, tail, timestamp))

    Returns:
        entry_without_timestamp: i.e., (head, relation, tail)

    """
    assert len(entry) == 4
    logging.debug(f"Removing timestamp from {entry} ...")
    entry_without_timestamp = entry[:-1]
    logging.info(f"Timestamp is removed from {entry}: {entry_without_timestamp}")

    return entry_without_timestamp


def split_by_possessive(name_entity: str) -> tuple[str, str]:
    """Separate name and entity from the given string.

    Args:
        name_entity: e.g., "tae's laptop"

    Returns:
        name: e.g., tae
        entity: e.g., laptop

    """
    logging.debug(f"spliting name and entity from {name_entity}")
    if "'s " in name_entity:
        name, entity = name_entity.split("'s ")
    else:
        name, entity = None, None

    return name, entity


def remove_posession(entity: str) -> str:
    """Remove name from the entity.

    Args:
        entity: e.g., bob's laptop

    Returns:
        e.g., laptop

    """
    return entity.split("'s ")[-1]


def seed_everything(seed: int) -> None:
    """Seed every randomness to seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_json(fname: str) -> dict:
    """Read json"""
    logging.debug(f"reading json {fname} ...")
    with open(fname, "r") as stream:
        return json.load(stream)


def write_json(content: dict, fname: str) -> None:
    """Write json"""
    logging.debug(f"writing json {fname} ...")
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> dict:
    """Read yaml."""
    logging.debug(f"reading yaml {fname} ...")
    with open(fname, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content: dict, fname: str) -> None:
    """write yaml."""
    logging.debug(f"writing yaml {fname} ...")
    with open(fname, "w") as stream:
        yaml.dump(content, stream, indent=2, sort_keys=False)


def write_pickle(to_pickle: object, fname: str):
    """Read pickle"""
    logging.debug(f"writing pickle {fname} ...")
    with open(fname, "wb") as stream:
        foo = pickle.dump(to_pickle, stream)
    return foo


def read_pickle(fname: str):
    """Read pickle"""
    logging.debug(f"reading pickle {fname} ...")
    with open(fname, "rb") as stream:
        foo = pickle.load(stream)
    return foo


def write_csv(content: list, fname: str) -> None:
    with open(fname, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(content)


def read_data(data_path: str) -> dict:
    """Read train, val, test spilts.

    Args:
        data_path: path to data.

    Returns:
        data: {'train': list of training obs,
            'val': list of val obs,
            'test': list of test obs}

    """
    logging.debug(f"reading data from {data_path} ...")
    data = read_json(data_path)
    logging.info(f"Succesfully read data {data_path}")

    return data


def load_questions(path: str) -> dict:
    """Load premade questions.

    Args:
        path: path to the question json file.

    """
    logging.debug(f"loading questions from {path}...")
    questions = read_json(path)
    logging.info(f"questions loaded from {path}!")

    return questions


def argmax(iterable):
    """argmax"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def get_duplicate_dicts(search: dict, target: list) -> list:
    """Find if there are duplicate dicts.

    Args:
        search: dict
        target: target list to look up.

    Returns:
        duplicates: a list of dicts or None

    """
    assert isinstance(search, dict)
    logging.debug("finding if duplicate dicts exist ...")
    duplicates = []

    for candidate in target:
        assert isinstance(candidate, dict)
        if set(search).issubset(set(candidate)):
            if all([val == candidate[key] for key, val in search.items()]):
                duplicates.append(candidate)

    logging.info(f"{len(duplicates)} duplicates were found!")

    return duplicates


def list_duplicates_of(seq, item) -> list:
    # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def rename_training_dirs(root_dir: str = "./training-results/"):
    old_dirs = []
    new_dirs = []
    for foo in glob(os.path.join(root_dir, "*")):
        bar = glob(os.path.join(foo, "*/*/*test*"))
        if len(bar) == 0:
            continue
        bar = bar[0]
        old_dir = "/".join(bar.split("/")[:-1])
        old_dirs.append(old_dir)
        hparams = read_yaml(os.path.join(old_dir, "hparams.yaml"))

        allow_random_human = hparams["allow_random_human"]
        allow_random_question = hparams["allow_random_question"]
        pretrain_semantic = hparams["pretrain_semantic"]
        varying_rewards = hparams["varying_rewards"]
        capacity = hparams["capacity"]["episodic"] + hparams["capacity"]["semantic"]
        question_prob = hparams["question_prob"]
        des_size = hparams["des_size"]
        seed = hparams["seed"]

        new_dir = (
            f"training-results/"
            f"allow_random_human={allow_random_human}_"
            f"allow_random_question={allow_random_question}_"
            f"pretrain_semantic={pretrain_semantic}_"
            f"varying_rewards={varying_rewards}_"
            f"des_size={des_size}_"
            f"capacity={capacity}_"
            f"question_prob={question_prob}_"
            f"seed={seed}"
        )
        new_dirs.append(new_dir)
        os.rename(old_dir, new_dir)

    for foo in glob(os.path.join(root_dir, "*/lightning_logs")):
        if len(os.listdir(foo)) == 0:
            dir_to_delete = os.path.dirname(foo)
            shutil.rmtree(dir_to_delete)


def is_running_notebook() -> bool:
    """See if the code is running in a notebook or not."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def positional_encoding(
    positions: int,
    dimensions: int,
    scaling_factor: float = 10000.0,
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate sinusoidal positional encoding.

    Parameters:
    positions (int): The number of positions in the sequence.
    dimensions (int): The dimension of the embedding vectors.
    scaling_factor (float): The scaling factor used in the sinusoidal functions.
    return_tensor (bool): If True, return a PyTorch tensor; otherwise, return a NumPy array.

    Returns:
    Union[np.ndarray, torch.Tensor]: A positional encoding in the form of either a NumPy array or a PyTorch tensor.
    """
    # Ensure the number of dimensions is even
    assert dimensions % 2 == 0, "The dimension must be even."

    # Initialize a matrix of position encodings with zeros
    pos_enc = np.zeros((positions, dimensions))

    # Compute the positional encodings
    for pos in range(positions):
        for i in range(0, dimensions, 2):
            pos_enc[pos, i] = np.sin(pos / (scaling_factor ** ((2 * i) / dimensions)))
            pos_enc[pos, i + 1] = np.cos(
                pos / (scaling_factor ** ((2 * (i + 1)) / dimensions))
            )

    # Return as PyTorch tensor if requested
    if return_tensor:
        return torch.from_numpy(pos_enc)

    return pos_enc
