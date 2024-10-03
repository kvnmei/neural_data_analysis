import numpy as np


def gini_coefficient(data):
    """Calculate the Gini coefficient of a numpy array."""
    data = np.sort(data)  # Sort the data in ascending order
    n = len(data)
    cumulative_data = np.cumsum(data)  # Calculate the cumulative sum of the sorted data
    sum_data = np.sum(data)

    # Calculate the Gini coefficient
    gini = (2 * np.sum((np.arange(1, n + 1) * data)) - (n + 1) * sum_data) / (
        n * sum_data
    )

    return gini
