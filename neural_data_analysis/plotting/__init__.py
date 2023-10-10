# from .plotting import *
from .plotting import plot_ecdf, scatter_with_images, variance_explained

__version__ = "0.0.1"
__description__ = "A package for plotting functions."
__all__ = [f for f in dir() if not f.startswith("_")]
