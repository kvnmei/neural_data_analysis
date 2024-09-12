from .audio_processing import extract_audio_from_video
from .nwb_tools import get_nwb_files
from .string_processing import (
    get_brain_area_abbreviation,
    remove_lateralization,
    create_filename_from_dict,
)
from .utils import (
    correct_filepath,
    create_order_index,
    recursive_dict_update,
)
from .logger_setup import setup_logger, setup_default_logger

__version__ = "0.0.1"
__description__ = "A package for miscellaneous functions."
__all__ = [f for f in dir() if not f.startswith("_")]
