import importlib
import warnings

import numpy as np
import pandas as pd
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


def accuracy_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_pred.shape) > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return (y_true == y_pred).mean()


def generate_random_result(base_path):
    result = make_result(
        base_path=base_path,
        config={
            "seed": np.random.randint(0, 1000),
            "bs": np.random.randint(128, 1024, 10).tolist(),
            "lr": np.random.rand(10).tolist(),
            "momentum": np.random.rand(10).tolist(),
            "estimator_config": {
                "hidden_layers": np.random.randint(1, 10, 10).tolist(),
                "hidden_units": np.random.randint(1, 1000, 10).tolist(),
                "activation": str(np.random.choice(["relu", "tanh", "sigmoid"])),
                "optimizer": str(np.random.choice(["adam", "sgd", "rmsprop"])),
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
    return [generate_random_result(result_path) for _ in range(10)]


@pytest.fixture
def result_set(result_list):
    return ResultSet(result_list)


@pytest.fixture
def dataframe_path(tmp_path):
    return tmp_path / "results.xls"


def test_result_set_init(result_set, result_list):
    assert len(result_set) == len(result_list)

    for r in result_list:
        assert r in result_set
        assert result_set.contains(r)
        assert result_set[r.config] == r

    results_dict = result_set.results_
    new_result_set = ResultSet(results_dict)
    assert len(new_result_set) == len(result_list)
    assert new_result_set.results_ == results_dict

    with pytest.raises(TypeError):
        ResultSet(None)

    with pytest.raises(TypeError):
        ResultSet(1)

    with pytest.raises(TypeError):
        ResultSet("")

    with pytest.raises(TypeError):
        ResultSet([1, 2, 3])


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

    with pytest.raises(TypeError):
        result_set.add(None)

    with pytest.raises(TypeError):
        result_set.add(1)

    with pytest.raises(TypeError):
        result_set.add("")

    with pytest.raises(TypeError):
        result_set.add([1])

    with pytest.raises(TypeError):
        result_set.remove(None)

    with pytest.raises(TypeError):
        result_set.remove(1)

    with pytest.raises(KeyError):
        result_set.remove("")

    with pytest.raises(TypeError):
        result_set.remove([1])


def test_result_set_contains(result_set, result_list):
    for r in result_list:
        assert r in result_set
        assert result_set.contains(r)

        assert r.config in result_set
        assert result_set.contains(r.config)

        assert str(sanitize_json(r.config)) in result_set
        assert result_set.contains(str(sanitize_json(r.config)))

    with pytest.raises(TypeError):
        result_set.contains(None)

    with pytest.raises(TypeError):
        result_set.contains(1)

    with pytest.raises(TypeError):
        result_set.contains([1])


def test_result_set_filter_by_config(result_set):
    config = {
        "estimator_config": {
            "optimizer": "adam",
            "activation": "relu",
        }
    }

    next(iter(result_set)).config["estimator_config"]["optimizer"] = "adam"
    next(iter(result_set)).config["estimator_config"]["activation"] = "relu"

    filtered_result_set = result_set.filter_by_config(config)
    assert len(filtered_result_set) <= len(result_set)
    assert all(
        r.config["estimator_config"]["optimizer"] == "adam"
        and r.config["estimator_config"]["activation"] == "relu"
        for r in filtered_result_set
    )
    assert len(filtered_result_set) > 0

    with pytest.raises(TypeError):
        result_set.filter_by_config(None)

    with pytest.raises(TypeError):
        result_set.filter_by_config(1)

    with pytest.raises(TypeError):
        result_set.filter_by_config([1])


def test_result_set_filter(result_set):
    def _filter_fn(result):
        return result.config["estimator_config"]["optimizer"] == "adam"

    def _filter_fn_error(result):
        return "invalid value"

    def _filter_fn_error2(result):
        return result.config["nonexistent"] == "test"

    for i, result in enumerate(result_set):
        if i == 0:
            result.config["estimator_config"]["optimizer"] = "adam"
        elif i == 1:
            result.config["estimator_config"]["optimizer"] = "sgd"

    filtered_result_set = result_set.filter(_filter_fn)
    assert len(filtered_result_set) <= len(result_set)
    assert all(
        r.config["estimator_config"]["optimizer"] == "adam" for r in filtered_result_set
    )
    assert len(filtered_result_set) > 0

    with pytest.raises(TypeError):
        result_set.filter(None)
    with pytest.raises(TypeError):
        result_set.filter([])
    with pytest.raises(TypeError):
        result_set.filter({})
    with pytest.raises(TypeError):
        result_set.filter(1)

    with pytest.raises(TypeError):
        result_set.filter(_filter_fn_error)

    with pytest.raises(KeyError):
        result_set.filter(_filter_fn_error2)


def test_result_set_str_repr(result_set):
    assert str(len(result_set)) in str(result_set)
    assert str(len(result_set)) in repr(result_set)
    assert type(result_set).__str__ is not object.__str__
    assert type(result_set).__repr__ is not object.__repr__


def test_result_set_getitem(result_set, result_list):
    for r in result_list:
        assert result_set[r.config] == r

    for r in result_set:
        assert result_set[r.config] == r
        assert result_set[r] == r

    with pytest.raises(TypeError):
        result_set[None]

    with pytest.raises(TypeError):
        result_set[1]

    with pytest.raises(KeyError):
        result_set[""]

    with pytest.raises(TypeError):
        result_set[[1]]


def test_result_folder_init(result_list, result_path):
    for r in result_list:
        r.save()

    result_folder = ResultFolder(result_path)
    assert len(result_folder) == len(result_list)
    assert all(r in result_folder for r in result_list)

    assert result_folder.base_path == result_path


def test_result_folder_load_error(result_list, result_path):
    for r in result_list:
        r.save()

    (result_list[0].base_path / f"{result_list[0].id}.pkl").unlink()

    with pytest.raises(FileNotFoundError, match="Could not find"):
        ResultFolder(result_path)

    (result_list[0].base_path / f"{result_list[0].id}.json").unlink()

    (result_list[1].base_path / f"{result_list[1].id}.json").unlink()

    with pytest.raises(UserWarning, match="Number of json"):
        warnings.filterwarnings("error")
        ResultFolder(result_path)


def test_result_set_create_dataframe(result_set, result_list, dataframe_path):
    for r in result_list:
        r.save()

    def _compute_metrics(targets, predictions):
        return {
            "accuracy": accuracy_score(targets, predictions),
            "mze": 1 - accuracy_score(targets, predictions),
        }

    df = result_set.create_dataframe(
        config_columns=[
            "estimator_config.hidden_layers",
            "estimator_config.optimizer",
            "lr",
            "momentum",
        ],
        metrics_fn=_compute_metrics,
        include_train=True,
        include_val=True,
        best_params_columns=["bs", "lr", "momentum"],
        n_jobs=1,
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )

    assert df is not None
    columns = df.columns
    assert "config_estimator_config.hidden_layers" in columns
    assert "config_estimator_config.optimizer" in columns
    assert "config_lr" in columns
    assert "config_momentum" in columns

    assert "accuracy" in columns
    assert "mze" in columns
    assert "train_accuracy" in columns
    assert "train_mze" in columns
    assert "val_accuracy" in columns
    assert "val_mze" in columns

    assert "best_bs" in columns
    assert "best_lr" in columns
    assert "best_momentum" in columns

    assert len(df) == len(result_set)

    df.to_excel(dataframe_path, index=False)
    assert dataframe_path.exists()
    assert dataframe_path.is_file()

    df_loaded = pd.read_excel(dataframe_path)
    assert df_loaded is not None
    assert isinstance(df_loaded, pd.DataFrame)
    assert len(df_loaded) == len(result_set)
    assert all(df_loaded.columns == df.columns)

    def _filter_fn(result):
        return result.config["estimator_config"]["optimizer"] == "adam"

    next(iter(result_set)).config["estimator_config"]["optimizer"] = "adam"

    df_filter = result_set.create_dataframe(
        config_columns=[
            "estimator_config.hidden_layers",
            "estimator_config.optimizer",
            "lr",
            "momentum",
        ],
        filter_fn=_filter_fn,
        metrics_fn=_compute_metrics,
        include_train=True,
        include_val=True,
        best_params_columns=["bs", "lr", "momentum"],
        n_jobs=1,
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )

    assert df_filter is not None
    assert len(df_filter) <= len(result_set)
    assert len(df_filter) <= len(df)
    assert len(df_filter) > 0

    with pytest.raises(ValueError):
        result_set.create_dataframe(
            config_columns=[
                "estimator_config.hidden_layers",
                "estimator_config.optimizer",
                "lr",
                "momentum",
            ],
            metrics_fn=_compute_metrics,
            include_train=True,
            include_val=True,
            best_params_columns=["bs", "lr", "momentum"],
            n_jobs=0,
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
        )

    result_set.create_dataframe(
        config_columns=[
            "estimator_config.hidden_layers",
            "estimator_config.optimizer",
            "lr",
            "momentum",
        ],
        metrics_fn=_compute_metrics,
        include_train=True,
        include_val=True,
        best_params_columns=["bs", "lr", "momentum", "non_existent"],
        n_jobs=1,
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )

    with pytest.raises(ValueError):
        result_set.create_dataframe(
            config_columns=[
                "estimator_config.hidden_layers",
                "estimator_config.optimizer",
                "lr",
                "momentum",
            ],
            metrics_fn=_compute_metrics,
            include_train=True,
            include_val=True,
            best_params_columns=["bs", "lr", "momentum"],
            n_jobs=1,
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
            raise_errors=True,
        )

    with pytest.raises(ValueError):
        result_set.create_dataframe(
            config_columns=[
                "estimator_config.hidden_layers",
                "estimator_config.optimizer",
                "lr",
                "momentum",
            ],
            metrics_fn=_compute_metrics,
            include_train=True,
            include_val=True,
            best_params_columns=["bs", "lr", "momentum"],
            n_jobs=1,
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
            raise_errors="not valid",
        )

    with pytest.raises(ValueError):
        result_set.create_dataframe(
            config_columns=[
                "estimator_config.hidden_layers",
                "estimator_config.optimizer",
                "lr",
                "momentum",
            ],
            metrics_fn=_compute_metrics,
            include_train=True,
            include_val=True,
            best_params_columns=["bs", "lr", "momentum"],
            n_jobs=1,
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
            raise_errors=False,
        )


def test_result_set_create_dataframe_nojoblib(result_set):
    def _compute_metrics(targets, predictions):
        return {
            "accuracy": accuracy_score(targets, predictions),
            "mze": 1 - accuracy_score(targets, predictions),
        }

    original_find_spec = importlib.util.find_spec

    def _find_spec(name, path=None, target=None):
        if name == "joblib":
            return None
        return original_find_spec(name, path, target)

    importlib.util.find_spec = _find_spec

    with pytest.raises(RuntimeWarning):
        warnings.filterwarnings("error")
        result_set.create_dataframe(
            config_columns=[
                "estimator_config.hidden_layers",
                "estimator_config.optimizer",
                "lr",
                "momentum",
            ],
            metrics_fn=_compute_metrics,
            include_train=True,
            include_val=True,
            best_params_columns=["bs", "lr", "momentum"],
            n_jobs=3,
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
        )
        warnings.resetwarnings()

    result_set.create_dataframe(
        config_columns=[
            "estimator_config.hidden_layers",
            "estimator_config.optimizer",
            "lr",
            "momentum",
        ],
        metrics_fn=_compute_metrics,
        include_train=True,
        include_val=True,
        best_params_columns=["bs", "lr", "momentum"],
        n_jobs=1,
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )

    warnings.filterwarnings("ignore")
    result_set.create_dataframe(
        config_columns=[
            "estimator_config.hidden_layers",
            "estimator_config.optimizer",
            "lr",
            "momentum",
        ],
        metrics_fn=_compute_metrics,
        include_train=True,
        include_val=True,
        best_params_columns=["bs", "lr", "momentum"],
        n_jobs=3,
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )
    warnings.resetwarnings()

    importlib.util.find_spec = original_find_spec


def test_result_set_create_dataframe_joblib(result_set):
    def _compute_metrics(targets, predictions):
        return {
            "accuracy": accuracy_score(targets, predictions),
            "mze": 1 - accuracy_score(targets, predictions),
        }

    result_set.create_dataframe(
        config_columns=[
            "estimator_config.hidden_layers",
            "estimator_config.optimizer",
            "lr",
            "momentum",
        ],
        metrics_fn=_compute_metrics,
        include_train=True,
        include_val=True,
        best_params_columns=["bs", "lr", "momentum"],
        n_jobs=3,
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )


def test_result_set_create_dataframe_custom_column_prefix(result_set, dataframe_path):
    def _compute_metrics(targets, predictions):
        return {
            "accuracy": accuracy_score(targets, predictions),
            "mze": 1 - accuracy_score(targets, predictions),
        }

    df = result_set.create_dataframe(
        config_columns=[
            "estimator_config.hidden_layers",
            "estimator_config.optimizer",
            "lr",
            "momentum",
        ],
        metrics_fn=_compute_metrics,
        include_train=True,
        include_val=True,
        best_params_columns=["bs", "lr", "momentum"],
        n_jobs=1,
        config_columns_prefix="c_",
        best_params_columns_prefix="b_",
    )

    assert df is not None
    columns = df.columns
    assert "c_estimator_config.hidden_layers" in columns
    assert "c_estimator_config.optimizer" in columns
    assert "c_lr" in columns
    assert "c_momentum" in columns

    assert "accuracy" in columns
    assert "mze" in columns
    assert "train_accuracy" in columns
    assert "train_mze" in columns
    assert "val_accuracy" in columns
    assert "val_mze" in columns

    assert "b_bs" in columns
    assert "b_lr" in columns
    assert "b_momentum" in columns

    assert len(df) == len(result_set)

    df.to_excel(dataframe_path, index=False)
    assert dataframe_path.exists()
    assert dataframe_path.is_file()

    df_loaded = pd.read_excel(dataframe_path)
    assert df_loaded is not None
    assert isinstance(df_loaded, pd.DataFrame)
    assert len(df_loaded) == len(result_set)
    assert all(df_loaded.columns == df.columns)
