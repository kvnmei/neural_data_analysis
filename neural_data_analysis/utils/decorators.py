def add_default_repr(cls):
    def _default_repr(self):
        # Default __repr__ implementation
        details = f"Class: {self.__class__.__name__}\n"
        details += "Attributes:\n"
        for key, value in self.__dict__.items():
            details += f"  {key}: {value!r} (type: {type(value).__name__})\n"
        methods = [
            method_name
            for method_name in dir(self)
            if callable(getattr(self, method_name)) and not method_name.startswith("_")
        ]
        details += "Methods:\n" + "\n".join(f"  {method}" for method in methods)
        return details

    # Store the default __repr__ in the class
    cls._default_repr = _default_repr
    # If the class doesn't define __repr__, use the default one
    if "__repr__" not in cls.__dict__:
        cls.__repr__ = _default_repr
    return cls
