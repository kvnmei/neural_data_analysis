from neural_data_analysis.analysis import Neuron
from neural_data_analysis.constants import brain_area_dict
from neural_data_analysis.utils import setup_default_logger
import numpy as np
import logging


class NeuralPopulation:
    def __init__(self, logger: logging.Logger = None):
        """
        Initialize the class
        """
        self.neurons: list[Neuron] = []
        if logger is None:
            self.logger = setup_default_logger()
        else:
            self.logger = logger

    def count_cells_by_region(self, brain_areas: list = []) -> tuple[list, list]:
        """
        Returns the brain areas and the number of cells in each area.

        Parameters:
            brain_areas (list): List of brain regions to count cells in.

        Returns:
            brain_area (list): List of brain areas.
            brain_area_cell_counts (list): List of number of cells in each brain area.
        """
        if not brain_areas:
            brain_areas = np.unique(
                [neuron.brain_area_abbreviation for neuron in self.neurons]
            )

        regions = []
        counts = []
        for ba in brain_areas:
            ba_cells = [
                cell
                for cell in self.neurons
                if any(
                    area in cell.brain_area_abbreviation for area in brain_area_dict[ba]
                )
            ]
            regions.append(ba)
            counts.append(len(ba_cells))
            self.logger.info(
                f"The number of cells in [{ba}] brain region(s) is [{len(ba_cells)}]."
            )
        return regions, counts
