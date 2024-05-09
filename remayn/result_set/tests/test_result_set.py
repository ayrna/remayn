import numpy as np
import pytest

from remayn.result import make_result
from remayn.result_set import ResultFolder, ResultSet
from remayn.utils.json import sanitize_json


class TestEstimator:
    __test__ = False

    def __init__(self, lr):
        self.lr = lr

    def __str__(self):
        return f"TestEstimator(lr={self.lr})"

    def __repr__(self):
        return str(self)

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


def generate_random_result(base_path):
    result = make_result(
        base_path=base_path,
        config={
            "bs": np.random.randint(128, 1024, 10).tolist(),
            "lr": np.random.rand(10).tolist(),
            "momentum": np.random.rand(10).tolist(),
            "estimator__config": {
                "hidden_layers": np.random.randint(1, 10, 10).tolist(),
                "hidden_units": np.random.randint(1, 1000, 10).tolist(),
                "activation": ["relu", "tanh", "sigmoid"],
                "optimizer": ["adam", "sgd", "rmsprop"],
                "loss": ["categorical_crossentropy", "mean_squared_error"],
                "loss_params": [{"reduction": "sum"}, {"reduction": "mean"}],
                "metrics": ["accuracy"],
            },
            "estimator": [TestEstimator(lr) for lr in np.random.rand(10)],
        },
        targets=np.random.randint(0, 10, 100),
        predictions=np.random.rand(100, 10),
        train_targets=np.random.randint(0, 10, 100),
        train_predictions=np.random.rand(100, 10),
        val_targets=np.random.randint(0, 10, 100),
        val_predictions=np.random.rand(100, 10),
        time=np.random.rand() * 1000,
        train_history=np.random.rand(100),
        val_history=np.random.rand(100),
        best_params={
            "bs": np.random.randint(128, 1024),
            "lr": np.random.rand(),
            "momentum": np.random.rand(),
        },
    )
    return result


@pytest.fixture
def result_path(tmp_path):
    return tmp_path / "results"


@pytest.fixture
def empty_result_set():
    return ResultSet([])


@pytest.fixture
def result_list(result_path):
    return [generate_random_result(result_path) for _ in range(100)]


@pytest.fixture
def result_set(result_list):
    return ResultSet(result_list)


def test_result_set_init(result_set, result_list):
    assert len(result_set) == 100

    for r in result_list:
        assert r in result_set
        assert result_set.contains(r)
        assert result_set[r.config] == r


def test_result_set_iter(result_set, result_list):
    for r1, r2 in zip(result_set, result_list):
        assert r1 == r2


def test_result_set_keys(result_set, result_list):
    for r in result_list:
        assert str(sanitize_json(r.config)) in result_set.results_.keys()

    for r in result_set:
        assert str(sanitize_json(r.config)) in result_set.results_.keys()
        assert result_set[r.config] == r


def test_result_set_add_remove(result_set, result_path):
    new_result = generate_random_result(result_path)
    result_set.add(new_result)

    assert new_result in result_set
    assert result_set.contains(new_result)
    assert result_set[new_result.config] == new_result
    assert str(sanitize_json(new_result.config)) in result_set.results_.keys()

    result_set.remove(new_result)

    assert new_result not in result_set
    assert not result_set.contains(new_result)
    assert str(sanitize_json(new_result.config)) not in result_set.results_.keys()

    new_result = generate_random_result(result_path)
    result_set.add(new_result)
    result_set.add(new_result)

    assert new_result in result_set
    assert result_set.contains(new_result)
    assert result_set[new_result.config] == new_result
    assert str(sanitize_json(new_result.config)) in result_set.results_.keys()

    result_set.remove(new_result.config)
    assert new_result not in result_set
    assert not result_set.contains(new_result)
    assert str(sanitize_json(new_result.config)) not in result_set.results_.keys()

    result_set.add(new_result)
    assert new_result in result_set
    assert result_set.contains(new_result)
    assert result_set[new_result.config] == new_result
    assert str(sanitize_json(new_result.config)) in result_set.results_.keys()

    result_set.remove(str(sanitize_json(new_result.config)))
    assert new_result not in result_set
    assert not result_set.contains(new_result)
    assert str(sanitize_json(new_result.config)) not in result_set.results_.keys()


def test_result_set_contains(result_set, result_list):
    for r in result_list:
        assert r in result_set
        assert result_set.contains(r)

        assert r.config in result_set
        assert result_set.contains(r.config)

        assert str(sanitize_json(r.config)) in result_set
        assert result_set.contains(str(sanitize_json(r.config)))


def test_result_set_str_repr(result_set):
    assert str(len(result_set)) in str(result_set)
    assert str(len(result_set)) in repr(result_set)
    assert type(result_set).__str__ is not object.__str__
    assert type(result_set).__repr__ is not object.__repr__


def test_result_folder_init(result_list, result_path):
    for r in result_list:
        r.save()

    result_folder = ResultFolder(result_path)
    assert len(result_folder) == len(result_list)
    assert all(r in result_folder for r in result_list)

    assert result_folder.base_path == result_path
