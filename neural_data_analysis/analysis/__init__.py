from .Electrode import Electrode
from .embedder_utils import embedder_from_spec, create_image_embeddings
from .Event import Event, SceneChange, NewOld, VideoFrame, SceneBoundary
from .Experiment import Experiment
from .ImageEmbedder import (
    CLIPEmbedder,
    ResNet50Embedder,
    ImageEmbedder,
    DINOEmbedder,
)

from .model_evaluation import (
    append_model_scores,
    evaluate_model_performance,
    calculate_model_performance,
    aggregate_metrics_across_iterations,
    average_across_iterations,
    sum_across_iterations,
    combine_across_iterations,
)
from .NeuralPopulation import NeuralPopulation
from .Neuron import Neuron
from .neural_population import (
    dataframe_from_cells,
    subset_cells,
)

from .single_neuron import calc_firing_rates

from .ResultsLoader import ResultsLoader
from .TextEmbedder import SGPTEmbedder, TextEmbedder
from .NLPProcessor import (
    NLPProcessor,
)
from .VideoLoader import VideoLoader

__version__ = "0.0.1"
__description__ = "A package for the data analysis pipeline of data validation, data wrangling, and hypothesis testing."
__all__ = [f for f in dir() if not f.startswith("_")]
