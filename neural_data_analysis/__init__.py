from .analysis import (
    CLIPEmbedder,
    Electrode,
    Event,
    LinearModel,
    MLPClassifier,
    MLPModel,
    MLPRegressor,
    Neuron,
    ResNet50Embedder,
    append_model_scores,
    calc_firing_rates,
    compute_psth,
    create_image_embeddings,
    dataframe_from_cells,
    embedder_config,
    embedder_from_spec,
    evaluate_model_performance,
    process_results_multiple_regression,
    subset_cells,
)
from .plotting import (
    model_performance_by_brain_area,
    model_performance_by_time,
    plot_ecdf,
    plot_model_predictions,
    plot_raster_psth,
    scatter_with_images,
    variance_explained,
)
from .utils import (
    average_across_iterations,
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
