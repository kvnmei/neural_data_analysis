"""
A set of string processing functions for working with neural data.

Functions:
    remove_lateralization
    get_brain_area_abbreviation
"""

from neural_data_analysis.constants import (
    brain_area_dict,
    brain_area_abbreviations_lower,
    brain_area_abbreviations_upper,
)


def remove_lateralization(name: str) -> str:
    """
    Removes the "Left" or "Right" designation in brain area name.

    Args:
        name (str): brain area name

    Returns:
        name (str): brain area name without "Left" or "Right"
    """
    if name.startswith("Left "):
        name = name.replace("Left ", "")
    elif name.startswith("Right "):
        name = name.replace("Right ", "")
    elif name.endswith("_left"):
        name = name.replace("_left", "")
    elif name.endswith("_right"):
        name = name.replace("_right", "")
    # elif name == "RSPE":
    #     print(f"Brain area: {name} has no lateralization.")
    else:
        raise ValueError(f"Brain area: {name} is not implemented here!")
    return name


def get_brain_area_abbreviation(name: str, lower=True) -> str:
    """
    Converts brain area name to its 3-letter abbreviation.

    Args:
        name (str): brain area name
        lower (bool): whether to return the abbreviation in lowercase

    Returns:
        abbreviated_name (str): 3-letter abbreviation of brain area name

    """
    if lower:
        abbreviated_name = brain_area_abbreviations_lower[name]
    else:
        abbreviated_name = brain_area_abbreviations_upper[name]
    return abbreviated_name


def create_filename_from_dict(prefix: str, config: dict, extension: str) -> str:
    """
    Create a filename from a dictionary of configuration parameters

    Args:
        prefix (str): Prefix to prepend to the filename
        config (dict): Dictionary of configuration parameters
        extension (str): File extension to append to the filename

    Returns:
        filename (str): Filename created from the configuration parameters
    """
    filename_parts = [prefix]

    for key, value in config.items():
        # Convert list or dict values to a string representation
        if isinstance(value, list):
            value_str = "_".join(map(str, value))
        elif isinstance(value, dict):
            value_str = "_".join(f"{k}-{v}" for k, v in value.items())
        else:
            value_str = str(value)

        # Replace any "/" with "_" in the value string
        value_str = value_str.replace("/", "_")

        # Append the key and value to the filename
        filename_parts.append(f"{key.upper()}_{value_str}")

    # Join all parts with an underscore
    filename = "_".join(filename_parts)

    # Add the file extension at the end
    filename += extension

    return filename
