from copy import deepcopy

import numpy as np
import pytest

from remayn.result import Result, ResultData, make_result


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


class WrongEstimator:
    def __init__(self, lr):
        self.lr = lr

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass


@pytest.fixture
def result_path(tmp_path):
    return tmp_path / "result"


@pytest.fixture
def result(result_path):
    return Result(result_path)


@pytest.fixture
def saved_result(result_path):
    return Result(result_path).save()


@pytest.fixture
def estimator():
    return TestEstimator(lr=1e-3)


@pytest.fixture
def result_config(estimator):
    return {
        "bs": [32, 64, 128],
        "estimator_config": {"lr": [1e-3, 1e-4]},
        "estimator": estimator,
    }


@pytest.fixture
def complete_result(result_path, result_config):
    return make_result(
        base_path=result_path,
        config=result_config,
        targets=np.array([1, 2, 3]),
        predictions=np.array([1, 2, 3]),
        train_targets=np.array([1, 2, 3]),
        train_predictions=np.array([1, 2, 3]),
        val_targets=np.array([1, 2, 3]),
        val_predictions=np.array([1, 2, 3]),
        time=1.0,
        train_history=np.array([1, 2, 3]),
        val_history=np.array([1, 2, 3]),
        best_params={"bs": 32, "estimator_config": {"lr": 1e-3}},
    ).save()


def test_init(result, result_path):
    assert result.base_path == result_path
    assert isinstance(result.id, str)
    assert result.config is None
    assert result.data_ is None
    assert result.data_md5sum_ is None
    assert result.created_at is None
    assert result.updated_at is None

    info_path = result_path / f"{result.id}.json"
    data_path = result_path / f"{result.id}.pkl"
    assert not info_path.exists()
    assert not data_path.exists()


def test_init_save(result, result_path):
    assert result.base_path == result_path
    assert isinstance(result.id, str)
    assert result.config is None
    assert result.data_ is None
    assert result.data_md5sum_ is None
    assert result.created_at is None
    assert result.updated_at is None

    info_path = result_path / f"{result.id}.json"
    data_path = result_path / f"{result.id}.pkl"
    assert not info_path.exists()
    assert not data_path.exists()

    result.save()
    assert info_path.exists()
    assert data_path.exists()


def test_make_result(complete_result, result_path, result_config):
    assert complete_result.base_path == result_path
    assert complete_result.config == result_config

    complete_result.load_data()
    data = complete_result.get_data()

    assert isinstance(data, ResultData)
    assert (data.targets == np.array([1, 2, 3])).all()
    assert (data.predictions == np.array([1, 2, 3])).all()
    assert (data.train_targets == np.array([1, 2, 3])).all()
    assert (data.train_predictions == np.array([1, 2, 3])).all()
    assert (data.val_targets == np.array([1, 2, 3])).all()
    assert (data.val_predictions == np.array([1, 2, 3])).all()
    assert data.time == 1.0
    assert (data.train_history == np.array([1, 2, 3])).all()
    assert (data.val_history == np.array([1, 2, 3])).all()
    assert data.best_params == {"bs": 32, "estimator_config": {"lr": 1e-3}}

    info_path = result_path / f"{complete_result.id}.json"
    data_path = result_path / f"{complete_result.id}.pkl"
    assert info_path.exists()
    assert data_path.exists()


def test_load_data(result, saved_result, complete_result, result_path, result_config):
    with pytest.raises(FileNotFoundError):
        result.load_data()

    saved_result.load_data()
    data = saved_result.get_data()
    assert data is None

    assert complete_result.base_path == result_path
    assert complete_result.config == result_config

    complete_result.load_data()
    data = complete_result.get_data()

    assert isinstance(data, ResultData)
    assert (data.targets == np.array([1, 2, 3])).all()
    assert (data.predictions == np.array([1, 2, 3])).all()
    assert (data.train_targets == np.array([1, 2, 3])).all()
    assert (data.train_predictions == np.array([1, 2, 3])).all()
    assert (data.val_targets == np.array([1, 2, 3])).all()
    assert (data.val_predictions == np.array([1, 2, 3])).all()
    assert data.time == 1.0
    assert (data.train_history == np.array([1, 2, 3])).all()
    assert (data.val_history == np.array([1, 2, 3])).all()
    assert data.best_params == {"bs": 32, "estimator_config": {"lr": 1e-3}}


def test_delete_result(result, saved_result, complete_result, result_path):
    # result object (not saved)
    info_path = result_path / f"{result.id}.json"
    data_path = result_path / f"{result.id}.pkl"

    assert not info_path.exists()
    assert not data_path.exists()

    with pytest.raises(FileNotFoundError):
        result.delete()
    result.delete(missing_ok=True)

    result.save()
    result.delete()

    assert not info_path.exists()
    assert not data_path.exists()

    # saved_result
    info_path = result_path / f"{saved_result.id}.json"
    data_path = result_path / f"{saved_result.id}.pkl"
    assert info_path.exists()
    assert data_path.exists()

    saved_result.delete()
    with pytest.raises(FileNotFoundError):
        saved_result.delete()
    saved_result.delete(missing_ok=True)

    assert not info_path.exists()
    assert not data_path.exists()

    # complete_result
    info_path = result_path / f"{complete_result.id}.json"
    data_path = result_path / f"{complete_result.id}.pkl"
    assert info_path.exists()
    assert data_path.exists()

    complete_result.delete()
    with pytest.raises(FileNotFoundError):
        complete_result.delete()
    complete_result.delete(missing_ok=True)

    assert not info_path.exists()
    assert not data_path.exists()


def test_load_result(saved_result, complete_result, result_path, result_config):
    # saved_result
    loaded_result = Result.load(result_path, saved_result.id)
    assert loaded_result.base_path == result_path
    assert loaded_result.config is None
    assert loaded_result.data_ is None
    assert loaded_result.data_md5sum_ is not None
    assert loaded_result.created_at is not None
    assert loaded_result.updated_at is not None

    info_path = result_path / f"{saved_result.id}.json"
    data_path = result_path / f"{saved_result.id}.pkl"
    assert info_path.exists()
    assert data_path.exists()

    # complete_result
    loaded_result = Result.load(result_path, complete_result.id)
    assert loaded_result.base_path == result_path
    assert loaded_result.compare_config(result_config)

    loaded_result.load_data()
    data = loaded_result.get_data()

    assert isinstance(data, ResultData)
    assert (data.targets == np.array([1, 2, 3])).all()
    assert (data.predictions == np.array([1, 2, 3])).all()
    assert (data.train_targets == np.array([1, 2, 3])).all()
    assert (data.train_predictions == np.array([1, 2, 3])).all()
    assert (data.val_targets == np.array([1, 2, 3])).all()
    assert (data.val_predictions == np.array([1, 2, 3])).all()
    assert data.time == 1.0
    assert (data.train_history == np.array([1, 2, 3])).all()
    assert (data.val_history == np.array([1, 2, 3])).all()
    assert data.best_params == {"bs": 32, "estimator_config": {"lr": 1e-3}}

    info_path = result_path / f"{complete_result.id}.json"
    data_path = result_path / f"{complete_result.id}.pkl"
    assert info_path.exists()
    assert data_path.exists()


def test_save_result(saved_result, complete_result, result_path, result_config):
    # saved_result
    saved_result.save()
    assert saved_result.config is None
    saved_result.config = {"bs": [32, 64, 128]}
    saved_result.save()
    loaded_result = Result.load(result_path, saved_result.id)
    assert loaded_result.config == {"bs": [32, 64, 128]}
    loaded_result.load_data()
    data = loaded_result.get_data()
    assert data is None
    loaded_result.set_data(
        ResultData(
            targets=np.array([1, 2, 3]),
            predictions=np.array([1, 2, 3]),
            train_targets=np.array([1, 2, 3]),
            train_predictions=np.array([1, 2, 3]),
            val_targets=np.array([1, 2, 3]),
            val_predictions=np.array([1, 2, 3]),
            time=1.0,
            train_history=np.array([1, 2, 3]),
            val_history=np.array([1, 2, 3]),
            best_params={"bs": 32, "estimator_config": {"lr": 1e-3}},
        )
    )
    loaded_result.save()
    loaded_result = Result.load(result_path, saved_result.id)
    loaded_result.load_data()
    data = loaded_result.get_data()
    assert isinstance(data, ResultData)
    assert (data.targets == np.array([1, 2, 3])).all()
    assert (data.predictions == np.array([1, 2, 3])).all()
    assert (data.train_targets == np.array([1, 2, 3])).all()
    assert (data.train_predictions == np.array([1, 2, 3])).all()
    assert (data.val_targets == np.array([1, 2, 3])).all()
    assert (data.val_predictions == np.array([1, 2, 3])).all()
    assert data.time == 1.0
    assert (data.train_history == np.array([1, 2, 3])).all()
    assert (data.val_history == np.array([1, 2, 3])).all()
    assert data.best_params == {"bs": 32, "estimator_config": {"lr": 1e-3}}

    # complete_result
    complete_result.save()
    assert complete_result.config == result_config
    complete_result.config = {"bs": [32, 64, 128]}
    complete_result.save()
    loaded_result = Result.load(result_path, complete_result.id)
    assert loaded_result.config == {"bs": [32, 64, 128]}
    loaded_result.load_data()
    data = loaded_result.get_data()
    assert isinstance(data, ResultData)
    assert (data.targets == np.array([1, 2, 3])).all()
    assert (data.predictions == np.array([1, 2, 3])).all()
    assert (data.train_targets == np.array([1, 2, 3])).all()
    assert (data.train_predictions == np.array([1, 2, 3])).all()
    assert (data.val_targets == np.array([1, 2, 3])).all()
    assert (data.val_predictions == np.array([1, 2, 3])).all()
    assert data.time == 1.0
    assert (data.train_history == np.array([1, 2, 3])).all()
    assert (data.val_history == np.array([1, 2, 3])).all()

    loaded_result.set_data(
        ResultData(
            targets=np.array([5, 2, 3]),
            predictions=np.array([6, 2, 0]),
            train_targets=np.array([7, 2, 3]),
            train_predictions=np.array([8, 2, 3]),
            val_targets=np.array([9, 2, 8]),
            val_predictions=np.array([0, 2, 3]),
            time=35.0,
            train_history=np.array([3, 5, 3]),
            val_history=np.array([2, 2, 3]),
            best_params={"bs": 64, "estimator_config": {"lr": 1e-5}, "new": 1},
        )
    )
    loaded_result.save()
    loaded_result = Result.load(result_path, complete_result.id)
    loaded_result.load_data()
    data = loaded_result.get_data()
    assert isinstance(data, ResultData)
    assert (data.targets == np.array([5, 2, 3])).all()
    assert (data.predictions == np.array([6, 2, 0])).all()
    assert (data.train_targets == np.array([7, 2, 3])).all()
    assert (data.train_predictions == np.array([8, 2, 3])).all()
    assert (data.val_targets == np.array([9, 2, 8])).all()
    assert (data.val_predictions == np.array([0, 2, 3])).all()
    assert data.time == 35.0
    assert (data.train_history == np.array([3, 5, 3])).all()
    assert (data.val_history == np.array([2, 2, 3])).all()
    assert data.best_params == {"bs": 64, "estimator_config": {"lr": 1e-5}, "new": 1}

    # try to assign and save a config with the WrongEstimator
    complete_result.config["estimator"] = WrongEstimator(lr=1e-3)
    with pytest.raises(ValueError):
        complete_result.save()


def test_result_comparison(result, saved_result, complete_result):
    assert result == saved_result
    assert result != complete_result
    assert saved_result != complete_result

    assert result == result
    assert saved_result == saved_result
    assert complete_result == complete_result

    assert result != 1
    assert saved_result != 1
    assert complete_result != 1

    assert result != "result"
    assert saved_result != "saved_result"
    assert complete_result != "complete_result"

    complete_result2 = deepcopy(complete_result)
    assert complete_result == complete_result2

    complete_result2.config = {"bs": [32, 64, 128]}
    assert complete_result != complete_result2
    complete_result2.config = complete_result.config
    assert complete_result == complete_result2

    # data_ is not compared
    data = complete_result2.get_data()
    data.targets = np.array([1, 1, 1])
    complete_result2.set_data(data)
    assert complete_result == complete_result2
    complete_result2.save()
    assert complete_result == complete_result2

    # It can be compared directly
    assert complete_result.get_data() != complete_result2.get_data()
