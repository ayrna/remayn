from typing import Callable

import numpy as np

from ..result import Result
from ..utils import get_deep_item_from_dict


def get_metric_columns_values(
    targets: np.ndarray,
    predictions: np.ndarray,
    prefix: str,
    metrics_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
):
    """Creates the row with the metrics values for the given targets and predictions.
    The name of each column is determined by appending the name of the metric to the
    given prefix. For example, if adding the training metrics, the prefix could be
    'train_', so that the columns are named 'train_<metric_name>'.

    Parameters
    ----------
    targets : np.ndarray
        The targets.
    predictions : np.ndarray
        The predictions.
    prefix : str
        The prefix to add to the column names.
    metrics_fn : Callable[[np.ndarray, np.ndarray], dict[str, float]]
        The function to calculate the metrics. See `ResultSet.get_dataframe` for more
        details.

    Returns
    -------
    row : dict
        The row with the metrics values. The keys represent the name of the columns and
        the values are the metrics values.
    """

    if targets is not None and predictions is not None:
        metrics = metrics_fn(targets, predictions)
        row = {f"{prefix}{column}": value for column, value in metrics.items()}
    else:
        row = {f"{prefix}{column}": None for column in metrics.keys()}

    return row


def get_row_from_result(
    result: Result,
    config_columns: list[str] = [],
    metrics_fn=lambda targets, predictions: {},
    include_train: bool = False,
    include_val: bool = False,
    best_params_columns: list[str] = [],
):
    targets = result.get_data().targets
    predictions = result.get_data().predictions
    time = result.get_data().time

    test_metrics = metrics_fn(targets, predictions)

    # Create row dict with config columns, best params columns and test metrics
    row = {
        **{
            column: get_deep_item_from_dict(result.get_config(), column)
            for column in config_columns
        },
        **{
            column: get_deep_item_from_dict(result.get_data().best_params, column)
            for column in best_params_columns
        },
        **test_metrics,
    }

    if include_train:
        train_metrics_row = get_metric_columns_values(
            result.get_data().train_targets,
            result.get_data().train_predictions,
            "train_",
            metrics_fn,
        )
        row = {**row, **train_metrics_row}

    if include_val:
        val_metrics_row = get_metric_columns_values(
            result.get_data().val_targets,
            result.get_data().val_predictions,
            "val_",
            metrics_fn,
        )
        row = {**row, **val_metrics_row}

    row["time"] = time

    # Add best epoch and loss value if histories are available
    if result.get_data().train_history is not None:
        row["best_train_epoch"] = result.get_data().train_history.argmin() + 1
        row["best_train_loss"] = result.get_data().train_history.min()

    if result.get_data().val_history is not None:
        row["best_val_epoch"] = result.get_data().val_history.argmin() + 1
        row["best_val_loss"] = result.get_data().val_history.min()

    return row
