from typing import Optional

import numpy as np


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
    ):
        if not isinstance(targets, np.ndarray):
            raise TypeError("targets must be a numpy array")
        if not isinstance(predictions, np.ndarray):
            raise TypeError("predictions must be a numpy array")
        if train_targets is not None and not isinstance(train_targets, np.ndarray):
            raise TypeError("train_targets must be a numpy array")
        if train_predictions is not None and not isinstance(
            train_predictions, np.ndarray
        ):
            raise TypeError("train_predictions must be a numpy array")
        if val_targets is not None and not isinstance(val_targets, np.ndarray):
            raise TypeError("val_targets must be a numpy array")
        if val_predictions is not None and not isinstance(val_predictions, np.ndarray):
            raise TypeError("val_predictions must be a numpy array")
        if time is not None and not isinstance(time, (int, float)):
            raise TypeError("time must be a number")
        if train_history is not None and not isinstance(train_history, np.ndarray):
            raise TypeError("train_history must be a numpy array")
        if val_history is not None and not isinstance(val_history, np.ndarray):
            raise TypeError("val_history must be a numpy array")
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
        )
