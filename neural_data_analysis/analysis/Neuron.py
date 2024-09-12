#!/usr/bin/env python3


# noinspection DuplicatedCode
class Neuron:
    """
    A class to represent a single neuron.
    """

    def __init__(self):
        self.id = None
        self.brain_area = None
        self.brain_area_abbreviation = None
        self.spike_times = None
        self.patient = None
        self.patient_session = None
        self.session = None

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
