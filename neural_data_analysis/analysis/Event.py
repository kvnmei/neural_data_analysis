"""

"""


class Event:
    def __int__(self):
        """
        Initialize the class
        """
        self.event_type = None
        self.event_time = None
        self.label = None
        self.duration = None

    def __repr__(self):
        """
        Return the class representation as a string.

        Returns:

        """
        # Class name
        details = f"Class: {self.__class__.__name__}\n"
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
