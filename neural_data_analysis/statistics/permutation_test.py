import numpy as np


# Function to perform permutation test
def permutation_test(
    group1: np.ndarray, group2: np.ndarray, n_permutations: int = 10000
) -> tuple[float, float]:
    """
    Permutation test for the difference in means between two groups.

    If the data were combined and randomly split into two groups over and over, how often would we see the observed difference in means?

    Parameters:
        group1 (np.ndarray): Values from group 1.
        group2 (np.ndarray): Values from group 2.
        n_permutations (int): The number of permutations of the combined group values to perform.

    Returns:

    """
    # Difference in means between the two groups
    observed_diff = np.abs(group1.mean() - group2.mean())

    # Combined the values in the two groups
    combined_values = np.concatenate([group1, group2])

    # Permute the values, split them in half, and calculate the difference in means
    perm_diffs = []
    for _ in range(n_permutations):
        np.random.shuffle(combined_values)
        perm_group1 = combined_values[: len(group1)]
        perm_group2 = combined_values[len(group1) :]
        perm_diff = np.abs(perm_group1.mean() - perm_group2.mean())
        perm_diffs.append(perm_diff)

    p_value = np.mean(perm_diffs >= observed_diff)
    return observed_diff, p_value
