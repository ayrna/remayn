import warnings
from copy import deepcopy

import numpy as np
import pytest

from remayn.result import make_result
from remayn.result_set.utils import get_metric_columns_values, get_row_from_result


def accuracy_score(targets, predictions):
    if len(predictions.shape) > 1:
        predictions = predictions.argmax(axis=1)
    if targets.shape[0] == 0:
        return 0
    return (targets == predictions).mean()


def metrics_fn(targets, predictions):
    return {
        "accuracy": accuracy_score(targets, predictions),
        "mze": 1 - accuracy_score(targets, predictions),
    }


def invalid_metrics_fn(targets, predictions):
    return accuracy_score(targets, predictions)


def generate_random_result(base_path, with_train=True, with_validation=True):
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
                "activation": np.random.choice(["relu", "tanh", "sigmoid"]),
                "optimizer": np.random.choice(["adam", "sgd", "rmsprop"]),
                "loss": ["categorical_crossentropy", "mean_squared_error"],
                "loss_params": [{"reduction": "sum"}, {"reduction": "mean"}],
                "metrics": ["accuracy"],
            },
        },
        targets=np.random.randint(0, 10, 100),
        predictions=np.random.rand(100, 10),
        train_targets=np.random.randint(0, 10, 100) if with_train else None,
        train_predictions=np.random.rand(100, 10) if with_train else np.array(None),
        val_targets=np.random.randint(0, 10, 100) if with_validation else np.array([]),
        val_predictions=np.random.rand(100, 10) if with_validation else [],
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
def result():
    return generate_random_result("./results")


@pytest.fixture
def result_without_trainval():
    return generate_random_result("./results", with_train=False, with_validation=False)


@pytest.fixture
def targets():
    return np.random.randint(0, 10, 500)


@pytest.fixture
def predictions():
    return np.random.rand(500, 10)


def test_get_metric_columns_values(targets, predictions):
    row = get_metric_columns_values(targets, predictions, "", metrics_fn)
    assert "accuracy" in row
    assert "mze" in row

    row = get_metric_columns_values(targets, predictions, "train_", metrics_fn)
    assert "train_accuracy" in row
    assert "train_mze" in row

    row = get_metric_columns_values(targets, predictions, "val_", metrics_fn)
    assert "val_accuracy" in row
    assert "val_mze" in row

    # Treat warnings as errors
    warnings.filterwarnings("error")

    with pytest.raises(UserWarning):
        get_metric_columns_values([], [], "", metrics_fn)

    with pytest.raises(UserWarning):
        get_metric_columns_values(None, None, "", metrics_fn)

    with pytest.raises(UserWarning):
        get_metric_columns_values(np.array(None), np.array(None), "", metrics_fn)

    with pytest.raises(UserWarning):
        get_metric_columns_values(np.array([]), np.array([]), "", metrics_fn)

    with pytest.raises(TypeError):
        get_metric_columns_values(
            {"this": "is a dict"}, {"this": "is a dict"}, "", metrics_fn
        )

    with pytest.raises(TypeError):
        get_metric_columns_values(targets, {"this": "is a dict"}, "", metrics_fn)

    with pytest.raises(TypeError):
        get_metric_columns_values({"this": "is a dict"}, predictions, "", metrics_fn)

    # Reset warnings
    warnings.resetwarnings()

    # Ignore warnings
    warnings.filterwarnings("ignore")

    row = get_metric_columns_values([], [], "", metrics_fn)
    assert row == {}

    row = get_metric_columns_values(None, None, "", metrics_fn)
    assert row == {}

    row = get_metric_columns_values(np.array(None), np.array(None), "", metrics_fn)
    assert row == {}

    row = get_metric_columns_values(np.array([]), np.array([]), "", metrics_fn)
    assert row == {}

    # Reset warnings
    warnings.resetwarnings()

    with pytest.raises(ValueError):
        get_metric_columns_values(targets, predictions, "", metrics_fn, raise_errors="")

    with pytest.raises(ValueError):
        get_metric_columns_values(
            targets, predictions, "", metrics_fn, raise_errors="invalid"
        )

    with pytest.raises(ValueError):
        get_metric_columns_values(
            targets, predictions, "", metrics_fn, raise_errors=False
        )

    # Treat warnings as errors
    warnings.filterwarnings("error")

    with pytest.raises(UserWarning):
        get_metric_columns_values(
            {"this": "is a dict"},
            {"this": "is a dict"},
            "",
            metrics_fn,
            raise_errors="warning",
        )

    with pytest.raises(UserWarning):
        get_metric_columns_values(
            targets,
            {"this": "is a dict"},
            "",
            metrics_fn,
            raise_errors="warning",
        )

    with pytest.raises(UserWarning):
        get_metric_columns_values(
            {"this": "is a dict"},
            predictions,
            "",
            metrics_fn,
            raise_errors="warning",
        )

    row = get_metric_columns_values(
        {"this": "is a dict"},
        {"this": "is a dict"},
        "",
        metrics_fn,
        raise_errors="ignore",
    )
    assert row == {}

    row = get_metric_columns_values(
        targets,
        {"this": "is a dict"},
        "",
        metrics_fn,
        raise_errors="ignore",
    )
    assert row == {}

    row = get_metric_columns_values(
        {"this": "is a dict"},
        predictions,
        "",
        metrics_fn,
        raise_errors="ignore",
    )
    assert row == {}

    # Reset warnings
    warnings.resetwarnings()

    # Ignore warnings
    warnings.filterwarnings("ignore")

    row = get_metric_columns_values(
        {"this": "is a dict"},
        {"this": "is a dict"},
        "",
        metrics_fn,
        raise_errors="warning",
    )
    assert row == {}

    row = get_metric_columns_values(
        targets,
        {"this": "is a dict"},
        "",
        metrics_fn,
        raise_errors="warning",
    )
    assert row == {}

    row = get_metric_columns_values(
        {"this": "is a dict"},
        predictions,
        "",
        metrics_fn,
        raise_errors="warning",
    )
    assert row == {}

    # Reset warnings
    warnings.resetwarnings()

    with pytest.raises(TypeError):
        get_metric_columns_values(targets, predictions, "", invalid_metrics_fn)


def test_get_row_from_result(result, result_without_trainval):
    get_row_from_result(
        result,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=True,
        include_val=True,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )

    new_result = deepcopy(result)
    new_result.data_ = None

    with pytest.raises(FileNotFoundError):
        get_row_from_result(
            new_result,
            config_columns=[
                "seed",
                "bs",
                "estimator_config.hidden_layers",
                "estimator_config.hidden_units",
            ],
            metrics_fn=metrics_fn,
            include_train=True,
            include_val=True,
            best_params_columns=[
                "bs",
                "lr",
                "momentum",
            ],
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
        )

    get_row_from_result(
        result,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=True,
        include_val=True,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix=None,
        best_params_columns_prefix=None,
    )

    result.data_.train_history = np.array(None)
    result.data_.val_history = np.array(None)
    get_row_from_result(
        result,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=True,
        include_val=True,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix=None,
        best_params_columns_prefix=None,
    )

    result.data_.train_history = np.array([])
    result.data_.val_history = np.array([])
    get_row_from_result(
        result,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=True,
        include_val=True,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix=None,
        best_params_columns_prefix=None,
    )

    with pytest.raises(TypeError):
        get_row_from_result(
            None,
            config_columns=[
                "seed",
                "bs",
                "estimator_config.hidden_layers",
                "estimator_config.hidden_units",
            ],
            metrics_fn=metrics_fn,
            include_train=True,
            include_val=True,
            best_params_columns=[
                "bs",
                "lr",
                "momentum",
            ],
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
        )

    with pytest.raises(TypeError):
        get_row_from_result(
            {"this": "is a dict"},
            config_columns=[
                "seed",
                "bs",
                "estimator_config.hidden_layers",
                "estimator_config.hidden_units",
            ],
            metrics_fn=metrics_fn,
            include_train=True,
            include_val=True,
            best_params_columns=[
                "bs",
                "lr",
                "momentum",
            ],
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
        )

    warnings.filterwarnings("error")

    with pytest.raises(UserWarning):
        get_row_from_result(
            result_without_trainval,
            config_columns=[
                "seed",
                "bs",
                "estimator_config.hidden_layers",
                "estimator_config.hidden_units",
            ],
            metrics_fn=metrics_fn,
            include_train=True,
            include_val=True,
            best_params_columns=[
                "bs",
                "lr",
                "momentum",
            ],
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
        )

    with pytest.raises(UserWarning):
        get_row_from_result(
            result_without_trainval,
            config_columns=[
                "seed",
                "bs",
                "estimator_config.hidden_layers",
                "estimator_config.hidden_units",
            ],
            metrics_fn=metrics_fn,
            include_train=False,
            include_val=True,
            best_params_columns=[
                "bs",
                "lr",
                "momentum",
            ],
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
        )

    with pytest.raises(UserWarning):
        get_row_from_result(
            result_without_trainval,
            config_columns=[
                "seed",
                "bs",
                "estimator_config.hidden_layers",
                "estimator_config.hidden_units",
            ],
            metrics_fn=metrics_fn,
            include_train=True,
            include_val=False,
            best_params_columns=[
                "bs",
                "lr",
                "momentum",
            ],
            config_columns_prefix="config_",
            best_params_columns_prefix="best_",
        )

    warnings.resetwarnings()

    warnings.filterwarnings("ignore")

    get_row_from_result(
        result_without_trainval,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=True,
        include_val=True,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )

    get_row_from_result(
        result_without_trainval,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=True,
        include_val=False,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )

    get_row_from_result(
        result_without_trainval,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=False,
        include_val=True,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
    )

    warnings.resetwarnings()

    get_row_from_result(
        result_without_trainval,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=True,
        include_val=True,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
        raise_errors="ignore",
    )

    get_row_from_result(
        result_without_trainval,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=True,
        include_val=False,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
        raise_errors="ignore",
    )

    get_row_from_result(
        result_without_trainval,
        config_columns=[
            "seed",
            "bs",
            "estimator_config.hidden_layers",
            "estimator_config.hidden_units",
        ],
        metrics_fn=metrics_fn,
        include_train=False,
        include_val=True,
        best_params_columns=[
            "bs",
            "lr",
            "momentum",
        ],
        config_columns_prefix="config_",
        best_params_columns_prefix="best_",
        raise_errors="ignore",
    )
