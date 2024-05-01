import json
import pickle
import time
from hashlib import md5
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

from ..utils import sanitize_json
from .result_data import ResultData


class Result:
    """Manages the result of a experiment.
    It contains the path where the experiment ResultData is stored, along with the
    experiment information.
    The ResultData is only loaded when needed to save memory and time.

    Attributes
    ----------
    base_path: str
        Base path where all the experiments are stored.
    id: str
        Unique identifier of the experiment.
    config_: dict
        Dictionary containing the parameters used in the experiment.
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
            id = str(uuid4())
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
        s = f"Config: {json.dumps(self.config, indent=4)}"
        if self.result_ is None:
            s += f"""
Results info path: {self.get_info_path()} (data not loaded)
"""
        else:
            s += f"""
Results info path: {self.get_info_path()}
Results data file: {self.get_data_path()}

Targets shape: {self.result_.targets.shape if self.result_.targets is not None else 'N/A'}
Predictions shape: {self.result_.predictions.shape if self.result_.predictions is not None else 'N/A'}
Train targets shape: {self.result_.train_targets.shape if self.result_.train_targets is not None else 'N/A'}
Train predictions shape: {self.result_.train_predictions.shape if self.result_.train_predictions is not None else 'N/A'}
Val targets shape: {self.result_.val_targets.shape if self.result_.val_targets is not None else 'N/A'}
Val predictions shape: {self.result_.val_predictions.shape if self.result_.val_predictions is not None else 'N/A'}

Time: {self.result_.time if self.result_.time is not None else 'N/A'}
Train history: {self.result_.train_history if self.result_.train_history is not None else 'N/A'}
Val history: {self.result_.val_history if self.result_.val_history is not None else 'N/A'}
Best params: {self.result_.best_params if self.result_.best_params is not None else 'N/A'}
"""
        return s

    def __repr__(self):
        return self.__str__()

    def load_data(self, force=False):
        """Load the ResultData from the disk.
        This method reads the ResultData from the disk and stores it in the result_
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

        if self.result_ is not None and not force:
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

        self.result_ = pickle.loads(content)

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
        result_ attribute directly.

        Parameters
        ----------
        force_reload: bool, optional, default=False
            If True, the ResultData will be reloaded even if it was already loaded.
            If False, it will only load the ResultData when it has not been loaded yet.

        Returns
        -------
        ResultData
            The ResultData object containing the results of the experiment.
        """

        if self.result_ is None or force_reload:
            self.load_data(force=force_reload)

        if self.result_ is None:
            raise FileNotFoundError(
                "ResultData could not be loaded. Make sure that the base_path is"
                " correct and the 'result_path' value in experiment_info_  dict is"
                " correct."
            )

        return self.result_

    def set_data(self, data: ResultData):
        """Sets the ResultData of the experiment.
        This method should be used to set the ResultData instead of setting the result_
        attribute directly.

        Parameters
        ----------
        data: ResultData
            The ResultData object containing the results of the experiment.
        """

        self.result_ = data

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

        # Save ResultData
        with open(data_path, "wb") as f:
            pickle.dump(self.result_, f)

        # Update md5sum based on new ResultData file
        with open(data_path, "rb") as f:
            self.data_md5sum_ = md5(f.read()).hexdigest()

        # Save experiment info
        safe_info = sanitize_json(self.get_experiment_info())
        with open(info_path, "w") as f:
            json.dump(safe_info, f, indent=4)

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
        """

        result = Result(base_path=base_path, id=id)

        info_path = result.get_info_path()

        with open(info_path, "r") as f:
            info = json.load(f)

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
    base_path,
    config,
    targets,
    predictions,
    train_targets=None,
    train_predictions=None,
    val_targets=None,
    val_predictions=None,
    time=None,
    train_history=None,
    val_history=None,
    best_params=None,
):
    """Helper function to create a `Result` object with the given data.
    It creates a `Result` object and the associated `ResultData`. The `Result` and
    the `ResultData` are not saved in the disk. You can call the `save()` method to
    save them.
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
        )
    )

    return result
