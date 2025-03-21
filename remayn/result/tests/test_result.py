import json
from copy import deepcopy
from hashlib import md5
from pathlib import Path

import numpy as np
import pytest

from remayn.result import Result, ResultData, make_result
from remayn.utils import sanitize_json


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

    def __eq__(self, other):
        return isinstance(other, TestEstimator) and self.lr == other.lr


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
        best_model=TestEstimator(lr=1e-3),
    ).save()


@pytest.fixture
def nonsaved_complete_result(result_path, result_config):
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
        best_model=TestEstimator(lr=1e-3),
    )


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
    assert data.best_model == TestEstimator(lr=1e-3)

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
    assert data.best_model == TestEstimator(lr=1e-3)

    complete_result.data_md5sum_ = "wrong_md5sum"
    with pytest.raises(ValueError):
        complete_result.load_data(force=True)


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
    assert data.best_model == TestEstimator(lr=1e-3)

    info_path = result_path / f"{complete_result.id}.json"
    data_path = result_path / f"{complete_result.id}.pkl"
    assert info_path.exists()
    assert data_path.exists()

    complete_result.get_info_path().unlink()
    with pytest.raises(FileNotFoundError):
        Result.load(result_path, complete_result.id)

    complete_result.get_info_path().write_text("wrong")
    with pytest.raises(json.JSONDecodeError):
        Result.load(result_path, complete_result.id)

    complete_result.get_info_path().write_text(json.dumps({"id": "wrong"}))
    with pytest.raises(ValueError, match="config"):
        Result.load(result_path, complete_result.id)

    complete_result.get_info_path().write_text(
        json.dumps(
            {"config": {"bs": [32, 64, 128]}},
        )
    )
    with pytest.raises(ValueError, match="md5"):
        Result.load(result_path, complete_result.id)


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
            best_model=TestEstimator(lr=1e-3),
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
    assert data.best_model == TestEstimator(lr=1e-3)

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
    assert data.best_params == {"bs": 32, "estimator_config": {"lr": 1e-3}}
    assert data.best_model == TestEstimator(lr=1e-3)

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
            best_model=TestEstimator(lr=1e-5),
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
    assert data.best_model == TestEstimator(lr=1e-5)

    # try to assign and save a config with the WrongEstimator
    complete_result.get_data_path().unlink(missing_ok=True)
    complete_result.get_info_path().unlink(missing_ok=True)
    complete_result.config["estimator"] = WrongEstimator(lr=1e-3)
    with pytest.raises(ValueError):
        complete_result.save()
    assert not complete_result.get_data_path().is_file()
    assert not complete_result.get_info_path().is_file()


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


def test_result_str_repr(result, saved_result, complete_result):
    assert "Results info path" in str(result)
    assert "not loaded" in str(result)
    assert repr(result) == str(result)

    assert "Results info path" in str(saved_result)
    assert "not loaded" in str(saved_result)
    assert repr(saved_result) == str(saved_result)

    assert "Results info path" in str(complete_result)
    assert "data file" in str(complete_result)
    assert repr(complete_result) == str(complete_result)


def test_result_compare_config(result, saved_result, complete_result):
    assert result.compare_config(saved_result)
    assert not result.compare_config(complete_result)
    assert not saved_result.compare_config(complete_result)

    assert result.compare_config(result)
    assert saved_result.compare_config(saved_result)
    assert complete_result.compare_config(complete_result)

    complete_result2 = deepcopy(complete_result)
    assert complete_result.compare_config(complete_result2)

    complete_result2.config = {"bs": [32, 64, 128]}
    assert not complete_result.compare_config(complete_result2)
    complete_result2.config = complete_result.config
    assert complete_result.compare_config(complete_result2)

    with pytest.raises(TypeError):
        complete_result.compare_config(1)


def test_result_copy_to(
    result, saved_result, complete_result, nonsaved_complete_result
):
    with pytest.raises(FileNotFoundError):
        result.copy_to(f"{result.base_path}_new")

    for r in [saved_result, complete_result, nonsaved_complete_result]:
        new_base_path = Path(f"{result.base_path}_new")
        r2 = r.copy_to(new_base_path)
        assert isinstance(r2, Result)
        assert r2.base_path == new_base_path
        assert r2.id == r.id
        assert sanitize_json(r2.config) == sanitize_json(r.config)
        assert r2 == r
        assert r.get_data() == r2.get_data()

        with open(r2.get_data_path(), "rb") as f:
            content = f.read()

        md5sum = md5(content).hexdigest()
        assert md5sum == r2.data_md5sum_

        if r.data_md5sum_ is not None:
            assert r.data_md5sum_ == r2.data_md5sum_

        assert new_base_path.exists()
        assert new_base_path / f"{r2.id}.json" == r2.get_info_path()
        assert new_base_path / f"{r2.id}.pkl" == r2.get_data_path()
        assert r2.get_info_path().exists()
        assert r2.get_data_path().exists()

        with pytest.raises(ValueError):
            r.copy_to(r.base_path)

        # Try to load the result from the new path
        loaded_result = Result.load(new_base_path, r.id)
        assert loaded_result.base_path == new_base_path
        assert sanitize_json(loaded_result.config) == sanitize_json(r.config)
        data = loaded_result.get_data()
        assert data == r.get_data()
