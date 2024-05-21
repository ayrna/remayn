from typing import Callable, Dict, List, Optional

import numpy as np

from ..result import Result
from ..utils import get_deep_item_from_dict


def get_metric_columns_values(
    targets: Optional[np.ndarray],
    predictions: Optional[np.ndarray],
    prefix: str,
    metrics_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
) -> Dict[str, float]:
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
    metrics_fn : Callable[[np.ndarray, np.ndarray], Dict[str, float]]
        The function to calculate the metrics. See `ResultSet.get_dataframe` for more
        details.

    Returns
    -------
    row : Dict[str, float]
        The row with the metrics values. The keys represent the name of the columns and
        the values are the metrics values.
    """

    if isinstance(targets, list):
        targets = np.array(targets)

    if isinstance(predictions, list):
        predictions = np.array(predictions)

    if targets is not None and predictions is not None:
        metrics = metrics_fn(targets, predictions)
        row = {f"{prefix}{column}": value for column, value in metrics.items()}
    else:
        row = {}

    return row


def get_row_from_result(
    result: Result,
    config_columns: List[str] = [],
    metrics_fn=lambda targets, predictions: {},
    include_train: bool = False,
    include_val: bool = False,
    best_params_columns: List[str] = [],
    config_columns_prefix: str = "config_",
    best_params_columns_prefix: str = "best_",
) -> Dict[str, float]:
    """Create a row with the information of a `Result`, which can be included in the
    pandas DataFrame of a `ResultSet`. The row contains the configuration columns
    provided, the best parameters columns provided, the test metrics, and optionally the
    training and validation metrics.

    A list containing several dictionaries returned by this function, can be used to
    create a pandas DataFrame with the information of several `Result` objects.

    Parameters
    ----------
    result : Result
        The `Result` object.
    config_columns : List[str], default=[]
        The names of the columns to include from the configuration.
    metrics_fn : Callable[[np.ndarray, np.ndarray], Dict[str, float]],
                default=lambda targets, predictions: {}
        See `ResultSet.get_dataframe` for more details.
    include_train : bool, default=False
        Whether to include the training metrics.
    include_val : bool, default=False
        Whether to include the validation metrics.
    best_params_columns : List[str], default=[]
        The names of the columns to include from the best parameters.
    config_columns_prefix : str, default="config_"
        The prefix to add to the configuration columns.
    best_params_columns_prefix : str, default="best_"
        The prefix to add to the best parameters columns.

    Returns
    -------
    row : Dict[str, float]
        The row with the information of the `Result`. Each key represents the name of a
        column and the value is the corresponding value.
    """

    data = result.get_data()

    if not config_columns_prefix:
        config_columns_prefix = ""

    if not best_params_columns_prefix:
        best_params_columns_prefix = ""

    targets = data.targets
    predictions = data.predictions
    time = data.time

    test_metrics = metrics_fn(targets, predictions)

    # Create row dict with config columns, best params columns and test metrics
    row = {
        **{
            f"{config_columns_prefix}{column}": get_deep_item_from_dict(
                result.config, column
            )
            for column in config_columns
        },
        **{
            f"{best_params_columns_prefix}{column}": get_deep_item_from_dict(
                data.best_params, column
            )
            for column in best_params_columns
        },
        **test_metrics,
    }

    if include_train:
        train_metrics_row = get_metric_columns_values(
            data.train_targets,
            data.train_predictions,
            "train_",
            metrics_fn,
        )
        row = {**row, **train_metrics_row}

    if include_val:
        val_metrics_row = get_metric_columns_values(
            data.val_targets,
            data.val_predictions,
            "val_",
            metrics_fn,
        )
        row = {**row, **val_metrics_row}

    row["time"] = time

    # Add best epoch and loss value if histories are available
    if data.train_history is not None:
        row["best_train_epoch"] = data.train_history.argmin() + 1
        row["best_train_loss"] = data.train_history.min()

    if data.val_history is not None:
        row["best_val_epoch"] = data.val_history.argmin() + 1
        row["best_val_loss"] = data.val_history.min()

    return row
