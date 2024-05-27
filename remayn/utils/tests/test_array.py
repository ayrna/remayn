import numpy as np
import pytest

from remayn.utils.array import check_array


def test_check_array():
    array = [1, 2, 3]
    assert check_array(array).tolist() == array
    assert (check_array(np.array(array)) == np.array(array)).all()

    array = None
    assert check_array(array, allow_none=True) is None
    with pytest.raises(TypeError):
        check_array(array)

    array = "string"
    with pytest.raises(TypeError):
        check_array(array)
