import numpy as np
import pytest

from remayn.result import Result, ResultData, make_result


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
def complete_result(result_path):
    return make_result(
        base_path=result_path,
        config={"bs": [32, 64, 128], "estimator_config": {"lr": [1e-3, 1e-4]}},
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


def test_make_result(complete_result, result_path):
    assert complete_result.base_path == result_path
    assert complete_result.config == {
        "bs": [32, 64, 128],
        "estimator_config": {"lr": [1e-3, 1e-4]},
    }

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


def test_load_data(result, saved_result, complete_result, result_path):
    with pytest.raises(FileNotFoundError):
        result.load_data()

    saved_result.load_data()
    data = saved_result.get_data()
    assert data is None

    assert complete_result.base_path == result_path
    assert complete_result.config == {
        "bs": [32, 64, 128],
        "estimator_config": {"lr": [1e-3, 1e-4]},
    }

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


def test_load_result(saved_result, complete_result, result_path):
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
    assert loaded_result.config == {
        "bs": [32, 64, 128],
        "estimator_config": {"lr": [1e-3, 1e-4]},
    }

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


def test_save_result(saved_result, complete_result, result_path):
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
    assert complete_result.config == {
        "bs": [32, 64, 128],
        "estimator_config": {"lr": [1e-3, 1e-4]},
    }
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
