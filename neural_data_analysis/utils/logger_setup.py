import logging
import logging.config
import yaml
from pathlib import Path


def setup_logger(
    logger_name: str = "logger", log_filepath: Path = Path("console.log")
) -> logging.Logger:
    """
    Set up a custom logger with a file handler and a console handler.

    Parameters:
        logger_name (str): Name of the logger.
        log_filepath (Path): Path to the log file.

    Returns:
        logging.Logger: Configured logger with file and stream handlers.
    """
    # Create a custom logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_filepath)

    # Set levels for handlers
    console_handler.setLevel(logging.INFO)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def setup_default_logger():
    """
    Set up a default logger with a console handler.

    Returns:
        logger (logging.Logger): A default logger with a console handler.
    """
    # Create a logger
    logger = logging.getLogger("DefaultLogger")
    logger.setLevel(logging.INFO)

    # Check if the logger already has handlers to avoid duplicate messages
    if not logger.hasHandlers():
        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create a formatter and set it for the handler
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger
