from .Electrode import Electrode
from .Event import Event
from .ImageEmbedder import (
    CLIPEmbedder,
    ResNet50Embedder,
    create_image_embeddings,
    embedder_config,
    embedder_from_spec,
)
from .LinearModel import LinearModel
from .MLPModel import MLPClassifier, MLPModel, MLPRegressor
from .Neuron import Neuron
from .model_evaluation import (
    append_model_scores,
    evaluate_model_performance,
    process_results_multiple_regression,
)
from .population import dataframe_from_cells, subset_cells
from .single_neuron import calc_firing_rates, compute_psth

__version__ = "0.0.1"
__description__ = "A package for the data analysis pipeline of data validation, data wrangling, and hypothesis testing."
__all__ = [f for f in dir() if not f.startswith("_")]
