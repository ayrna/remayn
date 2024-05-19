import json
import pickle
import time
from hashlib import md5
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import numpy as np

from ..utils import NonDefaultStrMethodError, sanitize_json
from .result_data import ResultData


class Result:
    """Represents the result of a experiment.
    It contains the path where the experiment ResultData is stored, along with the
    experiment information.
    The ResultData is only loaded when needed to save memory and time.

    Attributes
    ----------
    base_path: str
        Base path where all the experiments are stored.
    id: str
        Unique identifier of the experiment.
    config: dict
        Dictionary containing the parameters used in the experiment. All the elements
        in the dictionary must be JSON serializable. Objects contained in this dict
        should implement a custom __str__ method.
    data_: Optional[ResultData]
        Contains the `ResultData` when loaded or None if it was not loaded yet.
        This attribute should not be accessed directly. Use get_result() instead to
        make sure that the ResultData is properly loaded before accessing it.
    data_md5sum_: Optional[str]
        md5sum of the ResultData file. It is None if the file was not loaded yet or if
        creating a new Result that was not saved yet.
    created_at_: Optional[float]
        Timestamp when the experiment was created. It is None if the experiment was not
        saved yet.
    updated_at_: Optional[float]
        Timestamp when the experiment was last updated. It is None if the experiment was
        not saved yet.
    """

    base_path: Path
    id: str
    config: Optional[dict]
    data_: Optional[ResultData]
    data_md5sum_: Optional[str]
    created_at: Optional[float]
    updated_at: Optional[float]

    def __init__(
        self,
        base_path: Union[str, Path],
        id: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        """Initializes the Result object.
        By default, it does not load the whole ResultData.

        Parameters
        ----------
        base_path: Union[str, Path]
            Base path where all the experiments are stored.
        id: Optional[str]
            Unique identifier of the experiment. Will be used for the file names. If
            None, a new unique identifier will be generated.
        config: Optional[dict]
            Dictionary containing the parameters used in the experiment.
        """

        self.base_path = Path(base_path)
        if id is None:
            self.id = str(uuid4())
        else:
            self.id = id
        self.config = config
        self.data_ = None
        self.data_md5sum_ = None
        self.created_at = None
        self.updated_at = None

    def get_data_path(self):
        """Gets the path where the ResultData is stored.

        Returns
        -------
        Path
            Path where the ResultData is stored.
        """

        return self.base_path / f"{self.id}.pkl"

    def get_info_path(self):
        """Gets the path where the experiment information is stored.

        Returns
        -------
        Path
            Path where the experiment information is stored.
        """

        return self.base_path / f"{self.id}.json"

    def __str__(self):
        s = f"Config: {json.dumps(sanitize_json(self.config), indent=4)}"
        if self.data_ is None:
            s += f"""
Results info path: {self.get_info_path()} (data not loaded)
"""
        else:
            s += f"""
Results info path: {self.get_info_path()}
Results data file: {self.get_data_path()}

Targets shape: {self.data_.targets.shape if self.data_.targets is not None else 'N/A'}
Predictions shape: {self.data_.predictions.shape if self.data_.predictions is not None else 'N/A'}
Train targets shape: {self.data_.train_targets.shape if self.data_.train_targets is not None else 'N/A'}
Train predictions shape: {self.data_.train_predictions.shape if self.data_.train_predictions is not None else 'N/A'}
Val targets shape: {self.data_.val_targets.shape if self.data_.val_targets is not None else 'N/A'}
Val predictions shape: {self.data_.val_predictions.shape if self.data_.val_predictions is not None else 'N/A'}

Time: {self.data_.time if self.data_.time is not None else 'N/A'}
Train history: {self.data_.train_history if self.data_.train_history is not None else 'N/A'}
Val history: {self.data_.val_history if self.data_.val_history is not None else 'N/A'}
Best params: {self.data_.best_params if self.data_.best_params is not None else 'N/A'}
"""
        return s

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other: Union["Result", dict]):
        """Compares two `Result` objects considering only their config.
        It returns False if the type of the other object is not a `Result` or a dict.

        Parameters
        ----------
        other: Union[Result, dict]
            The other Result object or a dictionary containing the config of the
            other experiment.

        Returns
        -------
        bool
            True if the configs are equal and False otherwise.
        """

        if not isinstance(other, (Result, dict)):
            return False

        return self.compare_config(other)

    def compare_config(self, other: Union["Result", dict]) -> bool:
        """Compare the config of this Result with the config of `other` Result. It
        returns True if the configs are equal and False otherwise.

        Parameters
        ----------
        other: Union[Result, dict]
            The other Result object or a dictionary containing the config of the
            other experiment.

        Returns
        -------
        bool
            True if the configs are equal and False otherwise.
        """

        if isinstance(other, Result):
            return sanitize_json(self.config) == sanitize_json(other.config)
        elif isinstance(other, dict):
            return sanitize_json(self.config) == sanitize_json(other)
        else:
            raise TypeError(
                f"Expected a Result or a dict, but got {type(other)} instead."
            )

    def load_data(self, force=False):
        """Load the ResultData from the disk.
        This method reads the ResultData from the disk and stores it in the data_
        attribute. It also checks the integrity of the pickle file using the md5sum.
        This method is called automatically by get_result() when the ResultData is
        needed. However, you can call it manually to force the loading of the file.
        If the file was already loaded, this method does nothing, unless force=True is
        passed as an argument.

        Parameters
        ----------
        force: bool, optional, default=False
            If True, the file will be loaded even if it was already loaded.

        Raises
        ------
        FileNotFoundError
            If the ResultData does not exist.
        ValueError
            If the md5sum of the file does not match the one stored in the experiment
            information.
        """

        if self.data_ is not None and not force:
            return

        data_path = self.get_data_path()

        if not data_path.exists():
            raise FileNotFoundError(
                f"ResultData {data_path} does not exist."
                " The experiment is incomplete!"
            )

        with open(data_path, "rb") as f:
            content = f.read()

        md5sum = md5(content).hexdigest()

        if md5sum != self.get_md5sum():
            raise ValueError(
                f"ResultData {data_path} integrity check failed."
                " The file may have been modified after the experiment."
            )

        data = pickle.loads(content)
        self.data_ = data

    def get_md5sum(self):
        """Gets the md5sum of the ResultData file, which is stored in the experiment
        information.

        Returns
        -------
        str
            The md5sum of the ResultData file.
        """

        return self.data_md5sum_

    def get_data(self, force_reload=False):
        """Gets the ResultData of the experiment. If it was not loaded yet, it loads it
        from the disk. If the file was already loaded, it returns the stored object.
        This method should be used to access the ResultData instead of accessing the
        data_ attribute directly.

        Parameters
        ----------
        force_reload: bool, optional, default=False
            If True, the ResultData will be reloaded even if it was already loaded.
            If False, it will only load the ResultData when it has not been loaded yet.

        Returns
        -------
        ResultData
            The ResultData object containing the results of the experiment. None if the
            ResultData is empty.
        """

        if self.data_ is None or force_reload:
            self.load_data(force=force_reload)

        return self.data_

    def set_data(self, data: ResultData):
        """Sets the ResultData of the experiment.
        This method should be used to set the ResultData instead of setting the data_
        attribute directly.

        Parameters
        ----------
        data: ResultData
            The ResultData object containing the results of the experiment.
        """

        self.data_ = data

    def get_experiment_info(self):
        """Gets all the experiment info as a dictionary, including experiment config,
        timestamps, md5sum of the ResultData file and the path where the ResultData is
        stored.

        Returns
        -------
        dict
            Dictionary containing all the experiment information.
        """

        return {
            "config": self.config,
            "data_path": self.get_data_path(),
            "data_md5sum": self.get_md5sum(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def save(self):
        """Saves this `Result` to the disk.
        It saves the experiment information in the info_path file and the
        `ResultData` in the experiment_info_['results_path'] pickle file.
        If the files already exist, they will be overwritten.
        If the directory where the files should be saved does not exist, it will be
        created.

        Returns
        -------
        Result
            The `Result` object itself.
        """

        info_path = self.get_info_path()
        data_path = self.get_data_path()

        current_time = time.time()

        # Set time stamps
        if self.created_at is not None:
            self.updated_at = current_time
        else:
            self.created_at = current_time
            self.updated_at = current_time

        # Create directory
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Save ResultData
        with open(data_path, "wb") as f:
            pickle.dump(self.data_, f)

        # Update md5sum based on new ResultData file
        with open(data_path, "rb") as f:
            self.data_md5sum_ = md5(f.read()).hexdigest()

        # Save experiment info
        experiment_info = self.get_experiment_info()
        safe_info = None
        try:
            safe_info = sanitize_json(experiment_info, accept_default_str=False)
        except NonDefaultStrMethodError:
            raise ValueError(
                "Experiment info contains some fields that are subject to change when"
                " the experiment is loaded from disk. Please, make sure that all the"
                " elements within the config are JSON serializable and show a"
                f" deterministic representation.\n{safe_info=}"
            )

        with open(info_path, "w") as f:
            json.dump(safe_info, f, indent=4)

        return self

    def delete(self, missing_ok=False):
        """Deletes the experiment information file (json) and the ResultData file
        (pickle) from the disk.

        Parameters
        ----------
        missing_ok: bool, optional, default=False
            If True, the method will not raise an error if the files do not exist.

        Raises
        ------
        FileNotFoundError
            If the experiment information file or the `ResultData` file does not exist
            and `missing_ok` is False.

        Returns
        -------
        bool
            True if the files were deleted successfully.
        """

        info_path = self.get_info_path()
        results_path = self.get_data_path()

        info_path.unlink(missing_ok=missing_ok)
        results_path.unlink(missing_ok=missing_ok)

        return True

    @staticmethod
    def load(base_path: Union[str, Path], id: str) -> "Result":
        """Loads a `Result` from the disk.
        It loads the experiment information and the `ResultData` from the disk and
        creates a new `Result` object with the loaded data.

        Parameters
        ----------
        base_path: Union[str, Path]
            Base path where all the experiments are stored.
        id: str
            Unique identifier of the experiment.

        Returns
        -------
        Result
            A new `Result` object with the loaded data.

        Raises
        ------
        FileNotFoundError
            If the experiment information file does not exist.
        ValueError
            If the experiment information file is not a valid json file.
            If the experiment information file does not contain the 'config' key.
            If the experiment information file does not contain the 'data_md5sum' key.

        Examples
        --------
        >>> from remayn.result import Result
        >>> result = Result.load("./results", "123")
        """

        result = Result(base_path=base_path, id=id)

        info_path = result.get_info_path()

        try:
            with open(info_path, "r") as f:
                info = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Experiment information file {info_path} does not exist."
            )

        if "config" not in info:
            raise ValueError(
                f"Experiment information file {info_path} does not contain the 'config'"
                " key. It is not a valid experiment information file."
            )
        result.config = info["config"]

        if "data_md5sum" not in info:
            raise ValueError(
                f"Experiment information file {info_path} does not contain the"
                " 'data_md5sum' key. It is not a valid experiment information file."
            )
        result.data_md5sum_ = info["data_md5sum"]

        result.created_at = info["created_at"] if "created_at" in info else None
        result.updated_at = info["updated_at"] if "updated_at" in info else None

        return result


def make_result(
    base_path: Union[str, Path],
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
    best_model: Optional[object] = None,
):
    """Helper function to create a `Result` object with the given data.
    It creates a `Result` object and the associated `ResultData`. The `Result` and
    the `ResultData` are not saved in the disk. You can call the `save()` method to
    save them.

    Parameters
    ----------
    base_path: Union[str, Path]
        Path of the main directory that will contain this `Result` and all the other
        results related to these experiments.
    config: dict
        Dictionary containing the parameters used in the experiment.
    targets: np.ndarray
        Array containing the targets of the experiment. Any shape can be used.
    predictions: np.ndarray
        Array containing the predictions of the experiment. Any shape can be used.
    train_targets: Optional[np.ndarray], optional, default=None
        Array containing the training targets of the experiment. Any shape can be used.
    train_predictions: Optional[np.ndarray], optional, default=None
        Array containing the training predictions of the experiment. Any shape can be
        used.
    val_targets: Optional[np.ndarray], optional, default=None
        Array containing the validation targets of the experiment. Any shape can be
        used.
    val_predictions: Optional[np.ndarray], optional, default=None
        Array containing the validation predictions of the experiment. Any shape can be
        used.
    time: Optional[float], optional, default=None
        Time spent to run the experiment.
    train_history: Optional[np.ndarray], optional, default=None
        Array containing the training history recorded during the training process. It
        should be a 1D array with the value of the error on each iteration.
    val_history: Optional[np.ndarray], optional, default=None
        Array containing the validation history recorded during the training process. It
        should be a 1D array with the value of the error on each iteration.
    best_params: Optional[dict], optional, default=None
        Dictionary containing the best parameters found during the experiment. It can be
        used in case that the experiment employs a cross-validation process. It can be
        left as None if the experiment does not use a cross-validation process or the
        cross-validation process is splitted in different experiments.
    best_model: Optional[object], optional, default=None
        Best model found during the experiment.

    Returns
    -------
    Result
        A new `Result` object with the given data. The `Result` is not saved in the
        disk. You can call the `save()` method to save it.

    Examples
    --------
    >>> import numpy as np
    >>> from remayn.result import make_result
    >>> targets = np.array([1, 2, 3])
    >>> predictions = np.array([1.1, 2.2, 3.3])
    >>> config = {"model": "linear_regression"}
    >>> result = make_result("results", config, targets, predictions)
    >>> result.save()
    """

    # Create a new Result (empty id to create a new one)
    result = Result(base_path=base_path, config=config)

    result.set_data(
        ResultData(
            targets=targets,
            predictions=predictions,
            train_targets=train_targets,
            train_predictions=train_predictions,
            val_targets=val_targets,
            val_predictions=val_predictions,
            time=time,
            train_history=train_history,
            val_history=val_history,
            best_params=best_params,
            best_model=best_model,
        )
    )

    return result
