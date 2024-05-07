import numpy as np
import pytest

from remayn.result import make_result
from remayn.result_set import ResultSet


def generate_random_result(base_path):
    result = make_result(
        base_path=base_path,
        config={
            "bs": np.random.randint(128, 1024, 10).tolist(),
            "lr": np.random.rand(10).tolist(),
            "momentum": np.random.rand(10).tolist(),
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

    assert result_set.results_ == result_list
    for r1, r2 in zip(result_set, result_list):
        assert r1 == r2
