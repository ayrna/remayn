from typing import Optional

import numpy as np


class ResultFile:
    """Stores the results of a experiment.
    ResultFile objects only contain data and are usually stored as pickle files.

    Attributes
    ----------
    name: str
        Name of the experiment.
    config: dict
        Dictionary of parameters used in the experiment.
    targets: np.ndarray
        Numpy array of target values for the test set.
    predictions: np.ndarray
        Numpy array of predicted values for the test set.
    train_targets: Optional[np.ndarray]
        Numpy array of target values for the training set.
    train_predictions: Optional[np.ndarray]
        Numpy array of predicted values for the training set.
    val_targets: Optional[np.ndarray]
        Numpy array of target values for the validation set.
    val_predictions: Optional[np.ndarray]
        Numpy array of predicted values for the validation set.
    time: float
        Time taken to run the experiment.
    train_history: Optional[np.ndarray]
        Training history of the model, represented as the value of the error on each
        iteration.
    val_history: Optional[np.ndarray]
        Validation history of the model, represented as the value of the error on each
        iteration.
    best_params: Optional[dict]
        Dictionary of the best parameters found during the experiment.
        Can be used in case that the experiment employes a cross-validation process.
    """

    def __init__(
        self,
        name: str,
        *,
        config: dict,
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
        self.name = name
        self.config = config
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
