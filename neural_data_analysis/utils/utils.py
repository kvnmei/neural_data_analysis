#!/usr/bin/env python3
"""

Functions:
    recursive_dict_update(dict, dict):

Classes:
    None
"""

from pathlib import Path

import numpy as np
import pandas as pd


def recursive_dict_update(original_dict: dict, new_dict: dict) -> None:
    """
    Recursively update a dictionary in-place with another dictionary.
    new_dict should have the same hierarchy and keys as original_dict for the keys to be updated.

    Args:
        original_dict (dict): dictionary to be updated
        new_dict (dict): dictionary to update with

    """
    for key, value in new_dict.items():
        if isinstance(value, dict) and isinstance(original_dict.get(key), dict):
            recursive_dict_update(original_dict[key], value)
        else:
            original_dict[key] = value


# DEPRECATED in favor of pathlib.Path.parent
def correct_filepath(filepath: Path, project_dir: str) -> Path:
    """
    Adds a "../" to the filename if working from a script in project subdirectories
    or a "../../" if working from a script in project subdirectories/tests.

    Args:
        filepath (Path): path assuming that you are in the top-level project directory
        project_dir (str): name of the top-level project directory

    Returns:
        correct_path (Path): new path relative to where script is being run
    """
    if Path.cwd().parents[0] == project_dir:
        correct_path = filepath
    elif Path.cwd().parents[1] == project_dir:
        correct_path = ".." / filepath
    elif Path.cwd().parents[2] == project_dir:
        correct_path = "../.." / filepath
    else:
        raise ValueError(
            f"Current working directory is not in {project_dir}, {project_dir}/subdirectory, "
            f"or {project_dir}/subdirectory/tests."
        )
    return correct_path


def create_order_index(array1: list, array2: list):
    """
    Create an index to order array1 by array2.
    Values in the two arrays should match each other.

    Args:
        array1 (list): list of values to be ordered
        array2 (list): list of values to order by

    Example:
        index = create_order_index(array1, array2)
        array2 = [array1[i] for i in index]
        OR
        np.array(array2) == np.array(array1)[index]
    """
    assert set(array1) == set(
        array2
    ), "Values in the two arrays should match each other."
    index = []
    for i in np.arange(len(array2)):
        index.append(array1.index(array2[i]))

    return index


def subset_out_class_imbalance(df) -> pd.DataFrame:
    """
    Given a dataframe with class imbalance, subset out the minimum number of samples from each class.

    Returns:

    """
    # find the minimum numbe of samples in a class
    min_samples = min([len(df[df["category"] == cat]) for cat in all_cats])
    # subset out the minimum number of samples from each class
    df_balanced = pd.DataFrame()
    for cat in all_cats:
        df_cat = df[df["category"] == cat]
        df_balanced = pd.concat([df_balanced, df_cat.sample(min_samples)])
    # shuffle the dataframe
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)
    return df_balanced
