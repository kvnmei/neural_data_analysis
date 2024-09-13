from typing import List, Tuple

import numpy as np
from ..plotting import plot_heatmap_binary_matrix


def shift_binary_array(arr: np.ndarray, shift: int, plot: bool = False) -> np.ndarray:
    """
    Shifts a binary array by a specified number of positions.

    Args:
        arr (np.ndarray): binary array of 0s and 1s
        shift (int): number of positions to shift the array

    Returns:
        shifted_arr (np.ndarray): binary array shifted by the specified number of positions
    """
    shifted_arr = np.roll(arr, shift)

    if plot:
        matrix = np.vstack([arr, shifted_arr])
        plot_heatmap_binary_matrix(
            matrix,
            title="Original and Shifted Embeddings",
            xlabel="Frames",
            ylabel=None,
            ytick_labels=["Original", "Shifted"],
            plot_all_xtick_labels=False,
        )

    return shifted_arr


def shuffle_binary_array_by_group(
    arr: np.ndarray, seed: int = None, plot: bool = False, return_seed: bool = False
) -> np.ndarray:
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
    groups_original, total_zeros_original = find_consecutive_sequences_in_binary_array(
        arr
    )
    len_of_groups_original = [len(group) for group in groups_original]
    print(
        f"total groups of consecutive 1s before shuffling: {len(groups_original)}, total 0s: {total_zeros_original}"
    )
    # print(f"length of groups: {len_of_groups_original}")

    # shuffles the groups in place
    rng.shuffle(groups_original)

    zero_distribution = distribute_zeros(groups, total_zeros, seed)
    shuffled_arr = reconstruct_binary_array_from_groups(groups, zero_distribution)
    groups_shuffled, total_zeros_shuffled = find_consecutive_sequences_in_binary_array(
        shuffled_arr
    )
    len_of_groups_shuffled = [len(group) for group in groups_shuffled]
    print(
        f"total groups of consecutive 1s after shuffling: {len(groups)}, total 0s: {total_zeros}"
    )
    # print(f"length of groups: {len_of_groups_shuffled}")
    if plot:
        matrix = np.vstack([arr, shuffled_arr])
        plot_heatmap_binary_matrix(
            matrix,
            title="Original and Shuffled Embeddings",
            xlabel="Frames",
            ylabel=None,
            ytick_labels=["Original", "Shuffled"],
            plot_all_xtick_labels=False,
        )

    if return_seed:
        return shuffled_arr, seed
    else:
        return shuffled_arr


def find_consecutive_sequences_in_binary_array(arr) -> tuple[list[list[int]], int]:
    """
    For a binary array of 0s and 1s, finds groups of 1s.

    Args:
        arr (np.ndarray): binary array of 0s and 1s

    Returns:
        groups (list): list of lists of 1s
        total_zeros (int): total number of 0s in the array
    """
    groups = []
    current_group = []
    total_zeros = 0

    # goes through every value of the array
    for val in arr:
        if val == 1:
            if current_group:  # not empty, then append 1 to the current group
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