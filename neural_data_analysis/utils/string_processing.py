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
}

brain_area_dict = {
    "all": [
        "amygdala",
        "hippocampus",
        "orbitofrontal cortex",
        "anterior cingulate cortex",
        "supplementary motor area",
    ],
    "mtl": ["hippocampus", "amygdala"],
    "amygdala": ["amygdala", "Amygdala"],
    "amy": ["amygdala", "Amygdala"],
    "hippocampus": ["hippocampus", "Hippocampus"],
    "hpc": ["hippocampus", "Hippocampus"],
    "orbitofrontal cortex": ["orbitofrontal cortex"],
    "ofc": ["orbitofrontal cortex"],
    "anterior cingulate cortex": ["anterior cingulate cortex"],
    "acc": ["anterior cingulate cortex"],
    "ACC": ["anterior cingulate cortex"],
    "supplementary motor area": ["supplementary motor area"],
    "sma": ["supplementary motor area"],
    "presma": ["pre-supplementary motor area"],
    "vmpfc": ["ventromedial prefrontal cortex"],
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
