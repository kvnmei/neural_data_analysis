"""
Functions for population-level analyses.

Functions:
    subset_cells

"""

from typing import List

import numpy as np

from neural_data_analysis.analysis import Neuron


def subset_cells(cells: List[Neuron], area: str) -> List[Neuron]:
    """
    This function subsets out cells by brain region.

    Args:
        cells (list[Neuron]): a list of Neuron objects
        area (string): the brain area(s) to subset out

    Returns:
        cells_subset (list): a list of Neuron objects from the specified brain area(s)
    """
    brain_area_dict = {
        "all": [
            "amygdala",
            "hippocampus",
            "orbitofrontal cortex",
            "anterior cingulate cortex",
            "supplementary motor area",
        ],
        "mtl": ["hippocampus", "amygdala"],
        "amygdala": ["amygdala", "Amygdala"],
        "amy": ["amygdala", "Amygdala"],
        "hippocampus": ["hippocampus", "Hippocampus"],
        "hpc": ["hippocampus", "Hippocampus"],
        "orbitofrontal cortex": ["orbitofrontal cortex"],
        "ofc": ["orbitofrontal cortex"],
        "anterior cingulate cortex": ["anterior cingulate cortex"],
        "acc": ["anterior cingulate cortex"],
        "supplementary motor area": ["supplementary motor area"],
        "sma": ["supplementary motor area"],
    }
    subset_idx = []
    for i in np.arange(len(cells)):
        if cells[i].brain_area in brain_area_dict[area]:
            subset_idx.append(i)
    cells_subset = cells[subset_idx]
    return cells_subset
