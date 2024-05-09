import copy

import numpy as np


class NonDefaultStrMethodError(Exception):
    """Raised when an object does not have a custom __str__ method."""

    pass


def sanitize_json(data, accept_default_str=True):
    """Returns an object that can be safely serialized to json.

    Numpy arrays are converted to lists.
    Lists, tuples and dictionary are processed recursively.
    When no conversion is possible, the object is converted to a string.

    Parameters
    ----------
    data : Any
        The data to be sanitized.

    accept_default_str : bool, optional
        If True, the function will use the default string representation of the object
        even if it does not have a custom implementation of the __str__ method.
        If False, the function will raise an error if the object does not have a custom
        implementation of the __str__ method.
    Returns
    -------
    sanitized_data : Any
        The sanitized data.

    Raises
    ------
    ValueError
        If the data cannot be sanitized.
    """

    data = copy.deepcopy(data)

    if data is None:
        return None
    elif isinstance(data, (int, float, str, bool, str)):
        return data
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (list, tuple)):
        return [sanitize_json(d, accept_default_str) for d in data]
    elif isinstance(data, dict):
        return {k: sanitize_json(v, accept_default_str) for k, v in data.items()}
    elif hasattr(data, "__str__"):
        if accept_default_str or type(data).__str__ is not object.__str__:
            return str(data)
        else:
            raise NonDefaultStrMethodError(
                f"Object of type {type(data)} does not have a custom __str__ method."
                " Set accept_default_str to True to use the default __str__ method or"
                " implement a custom __str__ method."
            )
    else:
        raise ValueError(f"Cannot serialize data of type {type(data)} to json.")
