"""
A set of string processing functions for working with neural data.

Functions:
    remove_lateralization
    get_brain_area_abbreviation


"""

brain_area_shortnames = {
    "all": "all",
    "Amygdala": "amy",
    "amygdala": "amy",
    "Hippocampus": "hpc",
    "hippocampus": "hpc",
    "orbitofrontal cortex": "ofc",
    "anterior cingulate cortex": "acc",
    "ACC": "acc",
    "supplementary motor area": "sma",
    "preSMA": "presma",
    "vmPFC": "vmpfc",
    # "RSPE": "rspe",
}


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


def get_brain_area_abbreviation(name: str) -> str:
    """
    Converts brain area name to its 3-letter abbreviation.

    Args:
        name (str): brain area name

    Returns:
        abbreviated_name (str): 3-letter abbreviation of brain area name

    """

    abbreviated_name = brain_area_shortnames[name]
    return abbreviated_name
