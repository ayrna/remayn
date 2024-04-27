import copy

import numpy as np


def sanitize_json(data):
    """Returns an object that can be safely serialized to json.

    Numpy arrays are converted to lists.
    Lists, tuples and dictionary are processed recursively.
    When no conversion is possible, the object is converted to a string.

    Parameters
    ----------
    data : Any
        The data to be sanitized.

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
    elif isinstance(data, (int, float, str, bool)):
        return data
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (list, tuple)):
        return [sanitize_json(d) for d in data]
    elif isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif hasattr(data, "__str__"):
        return str(data)
    else:
        raise ValueError(f"Cannot serialize data of type {type(data)} to json.")
