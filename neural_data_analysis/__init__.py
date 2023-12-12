from .analysis import (
    append_model_scores,
    calc_firing_rates,
    CLIPEmbedder,
    create_image_embeddings,
    dataframe_from_cells,
    Electrode,
    embedder_config,
    embedder_from_spec,
    evaluate_model_performance,
    Event,
    LinearModel,
    MLPMultiClassClassifier,
    MLPBinaryClassifier,
    MLPModel,
    MLPRegressor,
    Neuron,
    process_results_multiple_regression,
    ResNet50Embedder,
    subset_cells,
)
from .plotting import (
    model_performance_by_brain_area,
    model_performance_by_time,
    pairwise_distance_heatmap,
    plot_confusion_matrix,
    plot_ecdf,
    plot_model_predictions,
    plot_object_detection_result,
    plot_scatter_with_images,
    plot_tsne_projection,
    plot_variance_explained,
    plot_raster_psth,
    compute_psth,
    compute_psth_per_category,
    plot_gantt_bar_chart,
)
from .utils import (
    average_across_iterations,
    correct_filepath,
    create_order_index,
    get_brain_area_abbreviation,
    get_nwb_files,
    recursive_dict_update,
    remove_lateralization,
)

__author__ = "Kevin Mei"
__email__ = "kmei@caltech.edu"
__version__ = "0.0.1"
__all__ = [f for f in dir() if not f.startswith("_")]
