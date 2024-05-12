import pytest

from remayn.utils import dict_contains_dict, get_deep_item_from_dict


@pytest.fixture
def test_dict():
    return {
        "a": {
            "b": {
                "c": 1,
            },
        },
        "d": {
            "e": 2,
        },
    }


def test_get_deep_item_from_dict(test_dict):
    assert get_deep_item_from_dict(test_dict, "a.b.c") == 1
    assert get_deep_item_from_dict(test_dict, "d.e") == 2
    assert get_deep_item_from_dict(test_dict, "a.b") == {"c": 1}
    assert get_deep_item_from_dict(test_dict, "a") == {"b": {"c": 1}}
    assert get_deep_item_from_dict(test_dict, "f") is None
    assert get_deep_item_from_dict(test_dict, "a.b.c.d") is None
    assert get_deep_item_from_dict(test_dict, "a.b.c", separator="/") is None
    assert get_deep_item_from_dict(test_dict, "a/b/c", separator="/") == 1


def test_dict_contains_dict(test_dict):
    assert dict_contains_dict(test_dict, {"a": {"b": {"c": 1}}})
    assert dict_contains_dict(test_dict, {"d": {"e": 2}})
    assert dict_contains_dict(test_dict, {"a": {"b": {"c": 1}}, "d": {"e": 2}})
    assert not dict_contains_dict(test_dict, {"a": {"b": {"c": 2}}})
    assert not dict_contains_dict(test_dict, {"a": {"b": {"c": 1}, "d": 1}})
    assert not dict_contains_dict(test_dict, {"a": {"b": 1}})
    assert not dict_contains_dict(test_dict, {"a": {"b": {"c": 1}, "d": 2}})
    assert not dict_contains_dict(test_dict, {"a": {"b": {"c": 1}, "d": 2}, "e": 3})
    assert not dict_contains_dict(
        test_dict, {"a": {"b": {"c": 1}, "d": 2}, "e": 3, "f": 4}
    )
    assert not dict_contains_dict(
        test_dict, {"a": {"b": {"c": 1}, "d": 2}, "e": 3, "f": 4, "g": 5}
    )
    assert not dict_contains_dict(
        test_dict, {"a": {"b": {"c": 1}, "d": 2}, "e": 3, "f": 4, "g": 5, "h": 6}
    )
    assert not dict_contains_dict(
        test_dict,
        {"a": {"b": {"c": 1}, "d": 2}, "e": 3, "f": 4, "g": 5, "h": 6, "i": 7},
    )
    assert not dict_contains_dict(
        test_dict,
        {"a": {"b": {"c": 1}, "d": 2}, "e": 3, "f": 4, "g": 5, "h": 6, "i": 7, "j": 8},
    )
