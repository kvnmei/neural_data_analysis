"""
Functions for population-level analyses.

Functions:
    subset_cells

"""


import numpy as np
import pandas as pd
from pandas import DataFrame
from typing import List

from neural_data_analysis.analysis import Neuron


def subset_cells(cells: np.array([Neuron]), attribute: str, value: str) -> List[Neuron]:
    """
    This function subsets out cells by brain region.

    Args:
        cells (np.array([Neuron])): a list of Neuron objects
        attribute (str): the attribute of the Neuron object to subset by
        value (str): the value of the attribute to subset by

    Returns:
        cells_subset (np.array): a list of Neuron objects with the specified attribute value(s)
    """

    subset_idx = []
    assert hasattr(cells[0], attribute)
    for i in np.arange(len(cells)):
        if attribute == "brain_area":
            if cells[i].brain_area in brain_area_dict[value]:
                subset_idx.append(i)
        elif getattr(cells[i], attribute) == value:
            subset_idx.append(i)

    cells_subset = cells[subset_idx]
    return cells_subset


def dataframe_from_cells(cells: np.array([Neuron])) -> DataFrame:
    cells_df = pd.DataFrame(
        {
            "cell_id": [cell.id for cell in cells],
            "brain_area": [cell.brain_area for cell in cells],
            "variant": [cell.variant for cell in cells],
            "patient": [cell.patient for cell in cells],
        }
    )
    return cells_df


brain_area_shortnames = {
    "all": "all",
    "Amygdala": "amy",
    "amygdala": "amy",
    "Hippocampus": "hpc",
    "hippocampus": "hpc",
    "orbitofrontal cortex": "ofc",
    "anterior cingulate cortex": "acc",
    "ACC": "acc",
    "supplementary motor area": "sma",
    "preSMA": "presma",
    "vmPFC": "vmpfc",
    # "RSPE": "rspe",
}

brain_area_dict = {
    "all": [
        "amy",
        "amygdala",
        "Amygdala",
        "AMY",
        "hpc",
        "hippocampus",
        "Hippocampus",
        "HPC",
        "ofc",
        "orbitofrontal cortex",
        "OFC",
        "acc",
        "ACC",
        "anterior cingulate cortex",
        "sma",
        "SMA",
        "supplementary motor area",
        "presma",
        "preSMA",
        "pre-supplementary motor area",
        "vmpfc",
        "vmPFC",
        "ventromedial prefrontal cortex",
    ],
    "mtl": [
        "amy",
        "AMY",
        "amygdala",
        "Amygdala",
        "hpc",
        "HPC",
        "hippocampus",
        "Hippocampus",
    ],
    "amygdala": ["amy", "AMY", "amygdala", "Amygdala"],
    "amy": ["amy", "AMY", "amygdala", "Amygdala"],
    "hippocampus": ["hpc", "HPC", "hippocampus", "Hippocampus"],
    "hpc": ["hpc", "HPC", "hippocampus", "Hippocampus"],
    "orbitofrontal cortex": ["orbitofrontal cortex", "ofc", "OFC"],
    "ofc": ["orbitofrontal cortex", "ofc", "OFC"],
    "anterior cingulate cortex": ["anterior cingulate cortex", "acc", "ACC"],
    "acc": ["anterior cingulate cortex", "acc", "ACC"],
    "ACC": ["anterior cingulate cortex", "acc", "ACC"],
    "supplementary motor area": ["supplementary motor area", "SMA", "sma"],
    "sma": ["supplementary motor area", "SMA", "sma"],
    "presma": ["pre-supplementary motor area", "preSMA", "presma"],
    "vmpfc": ["ventromedial prefrontal cortex", "vmPFC", "vmpfc"],
}
