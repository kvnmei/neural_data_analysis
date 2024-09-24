from .analysis import apply_notch_filter

__version__ = "0.0.1"
__description__ = "A package for working with EEG data."
__all__ = [f for f in dir() if not f.startswith("_")]
