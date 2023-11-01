from .Electrode import Electrode
from .Event import Event
from .ImageEmbedder import (CLIPEmbedder, create_image_embeddings, embedder_config, embedder_from_spec,
                            ResNet50Embedder)
from .LinearModel import LinearModel
from .model_evaluation import (
    append_model_scores,
    evaluate_model_performance,
    process_results_multiple_regression,
)
from .Neuron import Neuron
from .population import dataframe_from_cells, subset_cells
from .single_neuron import calc_firing_rates, compute_psth
from .MLPModel import MLPModel, MLPMultiClassClassifier, MLPBinaryClassifier, MLPRegressor

__version__ = "0.0.1"
__description__ = "A package for the data analysis pipeline of data validation, data wrangling, and hypothesis testing."
__all__ = [f for f in dir() if not f.startswith("_")]
