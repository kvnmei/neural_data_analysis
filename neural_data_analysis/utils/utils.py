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
from typing import List


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


def reshape_into_2d(arr: np.array) -> np.array:
    """
    Turns a numpy ndarray of 1-d shape (n,) into (n, 1) or keeps 2-d shape (n, m) into (n, m).

    Args:
        arr (np.ndarray): input array

    Returns:
        arr (np.ndarray): reshaped array
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ValueError("Array must be 1d or 2d.")
    return arr


def average_across_iterations(
    df: pd.DataFrame,
    iter_var: str,
    target_var: List[str],
    columns_to_keep: List[str] = (
        "brain_area",
        "bin_center",
        "bin_size",
        "embedding",
    ),
) -> pd.DataFrame:
    """
    Given a pandas DataFrame, average the target variable across all iterations.
    Assumes multiple iterations were run for the same set of parameters,
    and the same number of iterations were run for each set of parameters.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        iter_var (str): name of the column containing the iteration variable
        target_var (tuple[str]): name of the column(s) containing the target variable(s)
        columns_to_keep (tuple[str]): list of the original dataframe columns to keep in the new dataframe

    Returns:
        df_avg (pd.DataFrame): DataFrame containing the averaged data

    Example:
        A dataframe containing the results from a kfold cross-validation experiment.
        Average the correlation across all folds.
        df_avg = average_across_iterations(df, iter_var="fold", target_var="correlation")
    """

    #  dataframe conversion, for now just fixed by making it a string
    if isinstance(target_var, str):
        target_var = tuple(target_var)

    averaged_result_list_dict = []
    n_iter = len(np.unique(df[iter_var]))
    # WARNING: function will not work if the iter_var is not repeated consecutive order in the dataframe
    for i in np.arange(0, len(df), n_iter):
        _temp = df.iloc[i : i + n_iter].reset_index(drop=True)
        averaged_result_dict = {}
        for col in columns_to_keep:
            averaged_result_dict[f"{col}"] = _temp[f"{col}"][0]

        for var in target_var:
            variable_avg = np.mean(_temp[var].to_numpy(), axis=0)
            variable_std = np.std(_temp[var].to_numpy(), axis=0)
            averaged_result_dict.update(
                {
                    f"{var}_avg": variable_avg,
                    f"{var}_std": variable_std,
                }
            )
        averaged_result_list_dict.append(averaged_result_dict)

    averaged_result_df = pd.DataFrame(averaged_result_list_dict)
    return averaged_result_df
