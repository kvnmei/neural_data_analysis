#!/usr/bin/env python3
"""

Functions:
    recursive_dict_update(dict, dict):

Classes:
    None
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd


# from pathlib import Path


def recursive_dict_update(original_dict: Dict, new_dict: Dict) -> None:
    """
    Recursively update a dictionary in-place with another dictionary.
    new_dict should have the same hierarchy and keys as original_dict for the keys to be updated.

    Args:
        original_dict (Dict): dictionary to be updated
        new_dict (Dict): dictionary to update with

    """
    for key, value in new_dict.items():
        if isinstance(value, dict) and isinstance(original_dict.get(key), dict):
            recursive_dict_update(original_dict[key], value)
        else:
            original_dict[key] = value


def correct_filepath(filepath: str, project_dir: str) -> str:
    """
    Adds a "../" to the filename if working from a script in project subdirectories

    Args:
        filepath (str): path assuming that you are in the top-level project directory
        project_dir (str): name of the top-level project directory

    Returns:
        correct_path (str): new path relative to where script is being run
    """
    if os.getcwd().split("/")[-1] == project_dir:
        correct_path = filepath
    elif os.getcwd().split("/")[-2] == project_dir:
        correct_path = f"../{filepath}"
    else:
        raise ValueError(
            f"Current working directory is not in {project_dir} or {project_dir}/subdirectory."
        )
    return correct_path


def create_order_index(array1: List, array2: List):
    """
    Create an index to order array1 by array2.
    Values in the two arrays should match each other.

    Args:
        array1 (List): list of values to be ordered
        array2 (List): list of values to order by

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


def average_across_iterations(
    df: pd.DataFrame, iter_var: str, target_var: str
) -> pd.DataFrame:
    """
    Given a pandas DataFrame, average the target variable across all iterations.
    Assumes multiple iterations were run for the same set of parameters,
    and the same number of iterations were run for each set of parameters.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        iter_var (str): name of the column containing the iteration variable
        target_var (str): name of the column containing the target variable

    Returns:
        df_avg (pd.DataFrame): DataFrame containing the averaged data

    Example:
        A dataframe containing the results from a kfold cross-validation experiment.
        Average the correlation across all folds.
        df_avg = average_across_iterations(df, iter_var="fold", target_var="correlation")
    """
    df_avg = pd.DataFrame()
    n_iter = len(np.unique(df[iter_var]))
    for i in np.arange(0, len(df), n_iter):
        temp = df.iloc[i : i + n_iter].reset_index(drop=True)
        variable_avg = np.mean(temp[target_var].to_numpy())
        variable_std = np.std(temp[target_var].to_numpy())
        dict_avg = {
            "brain_area": temp["brain_area"][0],
            "bin_center": temp["bin_center"][0],
            "bin_size": temp["bin_size"][0],
            "embedding": temp["embedding"][0],
            f"{target_var}_avg": variable_avg,
            f"{target_var}_std": variable_std,
        }
        df_avg = pd.concat(
            [df_avg, pd.DataFrame(dict_avg, index=[0])],
            ignore_index=True,
        )
    return df_avg
