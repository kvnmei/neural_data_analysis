#!/usr/bin/env python3
from typing import Union
import pandas as pd
import polars as pl
import numpy as np
from .Event import Event


class Neuron:
    """
    A class to represent a single neuron.
    """

    def __init__(self):
        self.id: str = None
        self.brain_area: str = None
        self.brain_area_abbreviation: str = None
        self.brain_area_hemisphere: str = None
        self.mni_coordinates: dict = {}
        self.spike_times: np.ndarray = None
        self.patient: str = None
        self.patient_session: str = None
        self.session: str = None
        self.trials: Union[pd.DataFrame, pl.DataFrame] = None
        self.events: list[Event] = []

    def __repr__(self):
        # Class name
        details = f"Class: {self.__class__.__name__}"
        # Class attributes
        details += f"Attributes: \n"
        for key in self.__dict__:
            details += f"  {key}: {type(self.__dict__[key])}\n"
        # Class methods
        methods = [
            method
            for method in dir(self)
            if callable(getattr(self, method)) and not method.startswith("__")
        ]
        details += "Methods:\n" + "\n".join(f"  {method}" for method in methods)
        return details
