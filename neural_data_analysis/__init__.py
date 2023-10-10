from .analysis import (
    calc_firing_rates,
    compute_psth,
    Electrode,
    Event,
    ImageEmbedder,
    LinearModel,
    MLPClassifier,
    MLPRegressor,
    Neuron,
    subset_cells,
)
from .plotting import plot_ecdf, scatter_with_images, variance_explained
from .utils import (
    correct_filepath,
    create_order_index,
    get_brain_area_abbreviation,
    recursive_dict_update,
    remove_lateralization,
)

__author__ = "Kevin Mei"
__email__ = "kmei@caltech.edu"
__version__ = "0.0.1"
__all__ = [f for f in dir() if not f.startswith("_")]
