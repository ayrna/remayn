import warnings
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np

from ..result import Result
from ..utils import get_deep_item_from_dict


def get_metric_columns_values(
    targets: Optional[Union[np.ndarray, list]],
    predictions: Optional[Union[np.ndarray, list]],
    prefix: str,
    metrics_fn: Callable[[np.ndarray, np.ndarray], Dict[str, float]],
    raise_errors: Literal["error", "warning", "ignore"] = "error",
) -> Dict[str, float]:
    """Creates the row with the metrics values for the given targets and predictions.
    The name of each column is determined by appending the name of the metric to the
    given prefix. For example, if adding the training metrics, the prefix could be
    'train_', so that the columns are named 'train_<metric_name>'.

    Parameters
    ----------
    targets : np.ndarray or list
        The targets.
    predictions : np.ndarray or list
        The predictions.
    prefix : str
        The prefix to add to the column names.
    metrics_fn : Callable[[np.ndarray, np.ndarray], Dict[str, float]]
        The function to calculate the metrics. See `ResultSet.get_dataframe` for more
        details.
    raise_errors : Literal["error", "warning", "ignore"], default="error"
        If set to 'error', it will raise an error if the targets or predictions are not
        valid values. If set to 'warning', it will throw a warning instead. If set to
        'ignore', it will ignore the error and return an empty dictionary.

    Returns
    -------
    row : Dict[str, float]
        The row with the metrics values. The keys represent the name of the columns and
        the values are the metrics values.

    Raises
    ------
    ValueError
        If raise_errors is not 'error', 'warning' or 'ignore'.
    TypeError
        If the targets or predictions are not numpy arrays and raise_errors is 'error'.
    TypeError
        If the metrics function does not return a dictionary.
    """

    if raise_errors not in ["error", "warning", "ignore"]:
        raise ValueError(
            f"raise_errors must be 'error', 'warning' or 'ignore'."
            f" Found: {raise_errors}."
        )

    if isinstance(targets, list):
        targets = np.array(targets)

    if isinstance(predictions, list):
        predictions = np.array(predictions)

    if targets is not None and not isinstance(targets, np.ndarray):
        if raise_errors == "error":
            raise TypeError(
                f"If set, targets must be a numpy array. Found type: {type(targets)}."
                " You can set raise_errors='warning' to skip this error and throw a "
                "warning instead. You can also set raise_errors='ignore' to ignore "
                "this error."
            )
        else:
            targets = np.array([])
            if raise_errors == "warning":
                warnings.warn(
                    f"If set, targets must be a numpy array. Found type: {type(targets)}."
                    " Setting targets to an empty numpy array."
                )

    if predictions is not None and not isinstance(predictions, np.ndarray):
        if raise_errors == "error":
            raise TypeError(
                f"If set, predictions must be a numpy array."
                f" Found type: {type(predictions)}."
                " You can set raise_errors='warning' to skip this error and throw a "
                "warning instead. You can also set raise_errors='ignore' to ignore "
                "this error."
            )
        else:
            predictions = np.array([])
            if raise_errors == "warning":
                warnings.warn(
                    f"If set, predictions must be a numpy array."
                    f" Found type: {type(predictions)}."
                    " Setting predictions to an empty numpy array."
                )

    if (
        targets is not None
        and predictions is not None
        and len(targets.shape) > 0
        and len(predictions.shape) > 0
        and targets.shape[0] > 0
        and predictions.shape[0] > 0
    ):
        metrics = metrics_fn(targets, predictions)
        if not isinstance(metrics, dict):
            raise TypeError(
                f"metrics_fn must return a dictionary. Found type: {type(metrics)}."
            )
        row = {f"{prefix}{column}": value for column, value in metrics.items()}
    else:
        if raise_errors in ["error", "warning"]:
            warnings.warn(
                "Targets or predictions are empty. Skipping metrics calculation."
                f" Prefix: {prefix}"
            )
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
    raise_errors: Literal["error", "warning", "ignore"] = "error",
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
    raise_errors : Literal["error", "warning", "ignore"], default="error"
        Determines the behaviour when an error occurs during the calculation of the
        metrics. See `get_metric_columns_values` for more details.

    Returns
    -------
    row : Dict[str, float]
        The row with the information of the `Result`. Each key represents the name of a
        column and the value is the corresponding value.

    Raises
    ------
    TypeError
        If the input is not a `Result` object.
    """

    if not isinstance(result, Result):
        raise TypeError(f"result must be a Result object. Found type: {type(result)}.")

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
            raise_errors=raise_errors,
        )
        row = {**row, **train_metrics_row}

    if include_val:
        val_metrics_row = get_metric_columns_values(
            data.val_targets,
            data.val_predictions,
            "val_",
            metrics_fn,
            raise_errors=raise_errors,
        )
        row = {**row, **val_metrics_row}

    row["time"] = time

    # Add best epoch and loss value if histories are available
    if (
        isinstance(data.train_history, np.ndarray)
        and len(data.train_history.shape) > 0
        and len(data.train_history) > 0
    ):
        row["best_train_epoch"] = data.train_history.argmin() + 1
        row["best_train_loss"] = data.train_history.min()

    if (
        isinstance(data.val_history, np.ndarray)
        and len(data.val_history.shape) > 0
        and len(data.val_history) > 0
    ):
        row["best_val_epoch"] = data.val_history.argmin() + 1
        row["best_val_loss"] = data.val_history.min()

    return row
