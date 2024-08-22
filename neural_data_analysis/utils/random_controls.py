from typing import List, Tuple

import numpy as np


def randomize_binary_array_by_group(arr, seed=None, return_seed=False):
    """
    Randomizes the order of groups of 1s in a binary array. The number and size of the groups of 1s are preserved.

    Args:
        arr (np.ndarray): binary array of 0s and 1s
        seed (int): seed for random number generator
        return_seed (bool): whether to return the seed used for randomization

    Returns:
        randomized_arr (np.ndarray): binary array with groups of 1s in a random order
        seed (int): seed used for randomization
    """
    if seed is None:
        np.random.seed(seed)

    rng = np.random.default_rng(seed=seed)
    groups, total_zeros = find_consecutive_sequences_in_binary_array(arr)
    rng.shuffle(groups)
    zero_distribution = distribute_zeros(groups, total_zeros, seed)
    randomized_arr = reconstruct_binary_array_from_groups(groups, zero_distribution)

    if return_seed:
        return randomized_arr, seed
    else:
        return randomized_arr


def find_consecutive_sequences_in_binary_array(arr) -> tuple[list[list[int]], int]:
    """
    For a binary array of 0s and 1s, finds groups of 1s.

    Args:
        arr (np.ndarray): binary array of 0s and 1s

    Returns:
        groups (list): list of lists of 1s
    """
    groups = []
    current_group = []
    total_zeros = 0

    for val in arr:
        if val == 1:
            if current_group:  # not empty, then append
                current_group.append(1)
            else:
                current_group = [1]  # empty, then add the first 1
        else:
            total_zeros += 1
            if current_group:  # not empty, save the current and initialize a new group
                groups.append(current_group)
                current_group = []

    if current_group:  # Add the last group if it ends with 1s
        groups.append(current_group)

    return groups, total_zeros


def distribute_zeros(groups, total_zeros: int, seed: int = None):
    """
    Distributes zeros randomly among groups of 1s.

    Parameters:
        groups:
        total_zeros:
        seed:

    Returns:

    """
    if seed is None:
        np.random.seed(seed)

    rng = np.random.default_rng(seed=seed)
    # Number of places to insert zeros is one more than the number of groups
    zero_slots = len(groups) + 1
    # Randomly allocate zeros to these slots
    zero_distribution = [0] * zero_slots
    for _ in range(total_zeros):
        zero_distribution[rng.integers(0, zero_slots - 1)] += 1

    return zero_distribution


def reconstruct_binary_array_from_groups(
    groups: list[list], zero_distribution
) -> np.ndarray:
    """
    Reconstructs a binary array from groups of 1s.

    Args:
        groups (list): list of lists of 1s
        zero_distribution (list): list of number of zeros between groups of 1s

    Returns:
        np.ndarray: binary array with groups of 1s and 0s
    """
    result = []

    for zeros, group in zip(zero_distribution, groups):
        result.extend([0] * zeros)
        result.extend(group)

    # Add remaining zeros after the last group
    result.extend([0] * zero_distribution[-1])

    return np.array(result)
