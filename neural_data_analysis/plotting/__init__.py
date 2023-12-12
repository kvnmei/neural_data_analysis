from .model_results import (
    model_performance_by_brain_area,
    model_performance_by_time,
    plot_confusion_matrix,
    plot_model_predictions,
    plot_object_detection_result,
)
from .plotting import (
    pairwise_distance_heatmap,
    plot_ecdf,
    plot_scatter_with_images,
    plot_tsne_projection,
    plot_variance_explained,
)
from .raster_psth import (
    plot_raster_psth,
    compute_psth,
    plot_gantt_bar_chart,
    compute_psth_per_category,
)

__version__ = "0.0.1"
__description__ = "A package for plotting functions."
__all__ = [f for f in dir() if not f.startswith("_")]
