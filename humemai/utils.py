"""Utility functions."""

import csv
import json
import os
import pickle
import random
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import yaml


def remove_timestamp(entry: list[str]) -> list:
    """Remove the timestamp from a given observation/episodic memory.

    Args:
        entry: An observation / episodic memory in a quadruple format
            (i.e., (head, relation, tail, timestamp))

    Returns:
        entry_without_timestamp: i.e., (head, relation, tail)

    """
    assert len(entry) == 4
    entry_without_timestamp = entry[:-1]

    return entry_without_timestamp


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
    with open(fname, "r") as stream:
        return json.load(stream)


def write_json(content: dict, fname: str) -> None:
    """Write json"""
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> dict:
    """Read yaml."""
    with open(fname, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content: dict, fname: str) -> None:
    """write yaml."""
    with open(fname, "w") as stream:
        yaml.dump(content, stream, indent=2, sort_keys=False)


def write_pickle(to_pickle: object, fname: str):
    """Read pickle"""
    with open(fname, "wb") as stream:
        foo = pickle.dump(to_pickle, stream)
    return foo


def read_pickle(fname: str):
    """Read pickle"""
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
    data = read_json(data_path)

    return data


def load_questions(path: str) -> dict:
    """Load premade questions.

    Args:
        path: path to the question json file.

    """
    questions = read_json(path)

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
    duplicates = []

    for candidate in target:
        assert isinstance(candidate, dict)
        if set(search).issubset(set(candidate)):
            if all([val == candidate[key] for key, val in search.items()]):
                duplicates.append(candidate)

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


def merge_lists(lists) -> list:
    """Merge a list of lists of lists into a single list of lists.

    Deepcopy is used to avoid modifying the original lists / dicts.

    Args:
        lists: A list of lists of lists. Each sublist should have the format
            [key, value], where key is a tuple of three elements and value is a
            dictionary.

    Returns:
        merged_list: A list of lists with the format [key, value], where key is
            a tuple of three elements and value is a dictionary.
    """
    merged_dict = defaultdict(dict)

    for sublist in lists:
        key = tuple(sublist[:3])
        if key in merged_dict:
            # Merge dictionaries
            for k, v in sublist[3].items():
                if k in merged_dict[key]:
                    if isinstance(v, list):
                        # Merge lists and remove duplicates
                        merged_dict[key][k] = list(set(merged_dict[key][k] + v))
                    else:
                        # Handle non-list values
                        merged_dict[key][k] = max(merged_dict[key][k], v)
                else:
                    merged_dict[key][k] = deepcopy(v)
        else:
            merged_dict[key] = deepcopy(sublist[3])

    # Convert back to the original list of lists format
    merged_list = [[*k, v] for k, v in merged_dict.items()]

    return merged_list
