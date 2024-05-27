from typing import List, Union

import numpy as np


def check_array(array: Union[np.ndarray, List], allow_none=False) -> np.ndarray:
    """Check if the input is a numpy array. If not, convert it to a numpy array.

    Parameters
    ----------
    array : Union[np.ndarray, List]
        The input array to check.
    allow_none : bool, optional
        Whether to allow the input to be None, by default False

    Returns
    -------
    np.ndarray
        The input array as a numpy array.
    """

    if array is None:
        if allow_none:
            return None
        else:
            raise TypeError("None is not a valid array")
    if not isinstance(array, np.ndarray):
        if isinstance(array, list):
            return np.array(array)
        else:
            raise TypeError(f"{array} must be a numpy array")
    return array
