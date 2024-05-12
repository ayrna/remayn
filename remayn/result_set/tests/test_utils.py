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
                "activation": np.random.choice(["relu", "tanh", "sigmoid"]),
                "optimizer": np.random.choice(["adam", "sgd", "rmsprop"]),
                "loss": ["categorical_crossentropy", "mean_squared_error"],
                "loss_params": [{"reduction": "sum"}, {"reduction": "mean"}],
                "metrics": ["accuracy"],
            },
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
def result():
    return generate_random_result("./results")


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

    row = get_metric_columns_values([], [], "", metrics_fn)
    assert "accuracy" in row
    assert "mze" in row

    row = get_metric_columns_values(None, None, "", metrics_fn)
    assert row == {}


def test_get_row_from_result(result):
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
