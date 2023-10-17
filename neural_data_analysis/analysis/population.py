"""
Functions for population-level analyses.

Functions:
    subset_cells

"""


import numpy as np

from neural_data_analysis.analysis import Neuron
from neural_data_analysis.utils import brain_area_dict


def subset_cells(cells: np.array([Neuron]), area: str) -> list[Neuron]:
    """
    This function subsets out cells by brain region.

    Args:
        cells (np.array([Neuron])): a list of Neuron objects
        area (string): the brain area(s) to subset out

    Returns:
        cells_subset (list): a list of Neuron objects from the specified brain area(s)
    """

    subset_idx = []
    for i in np.arange(len(cells)):
        if cells[i].brain_area in brain_area_dict[area]:
            subset_idx.append(i)
    cells_subset = cells[subset_idx]
    return cells_subset
