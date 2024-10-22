from .analysis import (
    average_across_iterations,
    sum_across_iterations,
    embedder_from_spec,
    append_model_scores,
    calc_firing_rates,
    CLIPEmbedder,
    create_image_embeddings,
    dataframe_from_cells,
    DINOEmbedder,
    Electrode,
    evaluate_model_performance,
    Event,
    Experiment,
    SceneChange,
    SceneBoundary,
    NewOld,
    VideoFrame,
    LinearModelWrapper,
    MLPModelWrapper,
    NeuralPopulation,
    Neuron,
    calculate_model_performance,
    ResNet50Embedder,
    ResultsLoader,
    subset_cells,
    TextEmbedder,
    SGPTEmbedder,
    get_stem,
    get_synonyms_wordnet,
    create_word_groups,
    reduce_word_list_synonyms,
    create_excluded_words,
)

from .constants import (
    brain_area_dict,
    brain_area_abbreviations_lower,
    brain_area_abbreviations_upper,
    embedder_configs,
    gpt_configs,
)

from .LFP import (
    apply_notch_filter,
)

from .plotting import (
    model_performance_by_brain_area,
    model_performance_by_time,
    plot_heatmap_pairwise_distance,
    plot_confusion_matrix,
    plot_ecdf,
    plot_model_predictions,
    plot_object_detection_result,
    plot_scatter,
    plot_scatter_with_images,
    plot_tsne_projections,
    plot_variance_explained,
    # plot neural data
    plot_neuron_firing_rate,
    plot_raster_psth,
    compute_psth,
    compute_psth_per_category,
    plot_gantt_bar_chart,
    create_polar_plot_tuning_curve,
    plot_heatmap_binary_matrix,
    plot_heatmap_matrix,
)
from .statistics import (
    shift_binary_array,
    shuffle_binary_array_by_group,
    significance_stars,
    t_test,
    check_normality,
    check_homogeneity_of_variances,
    gini_coefficient,
    plot_data_distribution,
)
from .utils import (
    add_default_repr,
    correct_filepath,
    create_order_index,
    create_filename_from_dict,
    extract_audio_from_video,
    get_brain_area_abbreviation,
    get_nwb_files,
    recursive_dict_update,
    remove_lateralization,
    setup_logger,
    setup_default_logger,
)

__author__ = "Kevin J. M. Le"
__email__ = "kvnjmle@caltech.edu"
__version__ = "0.0.1"
f__all__ = [f for f in dir() if not f.startswith("_")]
