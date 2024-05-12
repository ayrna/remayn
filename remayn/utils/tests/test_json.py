import numpy as np
import pytest

from remayn.utils import NonDefaultStrMethodError, sanitize_json


class ExampleClass:
    def __str__(self):
        return "ExampleClass"

    def __repr__(self):
        return "ExampleClass"


class ExampleClassNoStr:
    def __repr__(self):
        return "ExampleClassNoStr"


@pytest.fixture
def test_json():
    return {
        "a": 1,
        "b": [1, 2, 3],
        "c": {"d": 4, "e": [5, 6, 7]},
        "f": None,
        "g": True,
        "h": False,
        "i": "test",
        "j": 1.0,
        "k": np.array([1, 2, 3]),
        "l": (1, 2, 3),
        "m": ExampleClass(),
        "n": ExampleClassNoStr(),
    }


def test_sanitize_json(test_json):
    sanitized_json = sanitize_json(test_json)
    assert sanitized_json == {
        "a": 1,
        "b": [1, 2, 3],
        "c": {"d": 4, "e": [5, 6, 7]},
        "f": None,
        "g": True,
        "h": False,
        "i": "test",
        "j": 1.0,
        "k": [1, 2, 3],
        "l": [1, 2, 3],
        "m": "ExampleClass",
        "n": "ExampleClassNoStr",
    }

    with pytest.raises(NonDefaultStrMethodError):
        sanitize_json(test_json, accept_default_str=False)

    original_hasattr = hasattr

    def overrided_hasattr(obj, attr):
        if attr == "__str__":
            return False
        return original_hasattr(obj, attr)

    builtins_hasattr = __builtins__["hasattr"]
    __builtins__["hasattr"] = overrided_hasattr
    with pytest.raises(ValueError):
        sanitize_json(test_json)
    __builtins__["hasattr"] = builtins_hasattr
