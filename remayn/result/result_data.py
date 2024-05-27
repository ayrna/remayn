from typing import Optional

import numpy as np

from ..utils import check_array


class ResultData:
    """Stores the results of a experiment.
    ResultData objects only contain data and are usually stored as pickle files.

    Attributes
    ----------
    targets: np.ndarray
        Numpy array of target values for the test set.
    predictions: np.ndarray
        Numpy array of predicted values for the test set.
    train_targets: Optional[np.ndarray], optional, default=None
        Numpy array of target values for the training set.
    train_predictions: Optional[np.ndarray], optional, default=None
        Numpy array of predicted values for the training set.
    val_targets: Optional[np.ndarray], optional, default=None
        Numpy array of target values for the validation set.
    val_predictions: Optional[np.ndarray], optional, default=None
        Numpy array of predicted values for the validation set.
    time: float, optional, default=None
        Time taken to run the experiment.
    train_history: Optional[np.ndarray], optional, default=None
        Training history of the model, represented as the value of the error on each
        iteration.
    val_history: Optional[np.ndarray], optional, default=None
        Validation history of the model, represented as the value of the error on each
        iteration.
    best_params: Optional[dict], optional, default=None
        Dictionary of the best parameters found during the experiment.
        Can be used in case that the experiment employes a cross-validation process.
    best_model: Optional[object], optional, default=None
        Best model found during the experiment.
    """

    def __init__(
        self,
        *,
        targets: np.ndarray,
        predictions: np.ndarray,
        train_targets: Optional[np.ndarray] = None,
        train_predictions: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
        val_predictions: Optional[np.ndarray] = None,
        time: Optional[float] = None,
        train_history: Optional[np.ndarray] = None,
        val_history: Optional[np.ndarray] = None,
        best_params: Optional[dict] = None,
        best_model: Optional[object] = None,
    ):
        targets = check_array(targets)
        predictions = check_array(predictions)
        train_targets = check_array(train_targets, allow_none=True)
        train_predictions = check_array(train_predictions, allow_none=True)
        val_targets = check_array(val_targets, allow_none=True)
        val_predictions = check_array(val_predictions, allow_none=True)
        train_history = check_array(train_history, allow_none=True)
        val_history = check_array(val_history, allow_none=True)

        if not isinstance(time, (float, int)) and time is not None:
            raise TypeError("time must be a float")

        if best_params is not None and not isinstance(best_params, dict):
            raise TypeError("best_params must be a dictionary")

        self.targets = targets
        self.predictions = predictions
        self.train_targets = train_targets
        self.train_predictions = train_predictions
        self.val_targets = val_targets
        self.val_predictions = val_predictions
        self.time = time
        self.train_history = train_history
        self.val_history = val_history
        self.best_params = best_params
        self.best_model = best_model

    def __eq__(self, other: "ResultData"):
        return (
            np.all(self.targets == other.targets)
            and np.all(self.predictions == other.predictions)
            and np.all(self.train_targets == other.train_targets)
            and np.all(self.train_predictions == other.train_predictions)
            and np.all(self.val_targets == other.val_targets)
            and np.all(self.val_predictions == other.val_predictions)
            and self.time == other.time
            and np.all(self.train_history == other.train_history)
            and np.all(self.val_history == other.val_history)
            and self.best_params == other.best_params
            and self.best_model == other.best_model
        )
