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
from neural_data_analysis.constants import brain_area_dict


def subset_cells(
    cells: np.array([Neuron]), attribute: str, value: str, logger=None
) -> List[Neuron]:
    """
    This function subsets out cells by brain region.

    Args:
        cells (np.array([Neuron])): a list of Neuron objects
        attribute (str): the attribute of the Neuron object to subset by
        value (str): the value of the attribute to subset by
        logger (logging.Logger): a logger object to log messages

    Returns:
        cells_subset (np.array): a list of Neuron objects with the specified attribute value(s)
    """
    if logger:
        logger.info(f"Subsetting cells by {attribute} = {value}")
    else:
        print(f"Subsetting cells by {attribute} = {value}")
    subset_idx = []
    assert hasattr(cells[0], attribute)
    for i in np.arange(len(cells)):
        if attribute == "brain_area":
            if cells[i].brain_area in brain_area_dict[value]:
                subset_idx.append(i)
            # DEBUGGING
            # else:
            #     print(cells[i].brain_area)
        elif getattr(cells[i], attribute) == value:
            subset_idx.append(i)

    cells_subset = cells[subset_idx]
    if logger:
        logger.info(
            f"Number of cells in full set: {len(cells)}. Number of cells in subset: {len(cells_subset)}"
        )
    else:
        print(
            f"Number of cells in full set: {len(cells)}. Number of cells in subset: {len(cells_subset)}"
        )
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
