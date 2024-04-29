import json
import pickle
import time
import warnings
from hashlib import md5
from pathlib import Path
from typing import Optional

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
    config_path: str
        Path to the file containing the experiment configuration. Relative to the
        base_path.
    experiment_info_: dict
        Dictionary containing information about the experiment. It includes the
        exact path of the ResultData inside the base_path directory (results_path).
        It also contains the md5sum of the file to check if it has been modified
        (results_md5sum).
    result_: Optional[ResultData]
        Contains the ResultData when loaded or None if it was not loaded yet.
        This attribute should not be accessed directly. Use get_result() instead to
        make sure that the ResultData is properly loaded before accessing it.
    load_time_: Optional[float]
        Time taken to load the ResultData. It is None if the file was not loaded yet.
    """

    base_path: str
    config_path: str
    experiment_info_: dict
    result_: Optional[ResultData]
    load_time_: Optional[float]

    def __init__(self, base_path: str, config_path: str, experiment_info: dict):
        """Initializes the Result object.
        By default, it does not load the whole ResultData.

        Parameters
        ----------
        base_path: str
            Base path where all the experiments are stored.
        config_path: str
            Path to the file containing the experiment configuration. Relative to the
            base_path.
        experiment_info: dict
            Dictionary containing information about the experiment. It should contain the
            exact path of the ResultData inside the base_path directory (results_path).
            It also contains the md5sum of the file to check if it has been modified
            (results_md5sum).

        Raises
        ------
        ValueError
            If the experiment_info dictionary does not contain the 'results_path' key.
        ValueError
            If the experiment_info dictionary does not contain the 'results_md5sum' key.
        """

        self.base_path = base_path
        self.config_path = config_path
        self.experiment_info_ = experiment_info
        self.result_ = None
        self.load_time_ = None

        if "results_path" not in experiment_info:
            raise ValueError(
                "The experiment_info dictionary must contain the 'results_path' key."
            )

        if "results_md5sum" not in experiment_info:
            raise ValueError(
                "The experiment_info dictionary must contain the 'results_md5sum' key."
            )

    def __str__(self):
        s = f"Config: {json.dumps(self.get_config(), indent=4)}"
        if self.result_ is None:
            s += f"""
Results info file: {self.config_path}
Results data file: {self.experiment_info_['results_path']} (not loaded)
"""
        else:
            s += f"""
Results info file: {self.config_path}
Results data file: {self.experiment_info_['results_path']}
Load time: {self.load_time_}

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

    def load(self, force=False):
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

        results_path = Path(self.base_path) / self.experiment_info_["results_path"]

        if not results_path.exists():
            raise FileNotFoundError(
                f"ResultData {results_path} does not exist."
                " The experiment is incomplete!"
            )

        start_time = time.time()

        with open(results_path, "rb") as f:
            content = f.read()

        md5sum = md5(content).hexdigest()

        if md5sum != self.experiment_info_["results_md5sum"]:
            raise ValueError(
                f"ResultData {results_path} integrity check failed."
                " The file may have been modified after the experiment."
            )

        self.result_ = pickle.loads(content)

        self.load_time_ = time.time() - start_time

    def get_config(self):
        """Gets the config of the experiment from the experiment_info_ dictionary.

        Returns
        -------
        dict
            Dictionary containing the parameters used in the experiment.
        """

        if "config" in self.experiment_info_:
            return self.experiment_info_["config"]
        else:
            warnings.warn(
                "'config' not found in experiment_info_ dictionary."
                "Storing the experiment config in the ResultData is deprecated"
                " and will be removed in future versions."
            )
            return self.get_data().config

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
            self.load(force=force_reload)

        if self.result_ is None:
            raise FileNotFoundError(
                "ResultData could not be loaded. Make sure that the base_path is"
                " correct and the 'result_path' value in experiment_info_  dict is"
                " correct."
            )

        return self.result_

    def save(self):
        """Saves this `Result` to the disk.
        It saves the experiment configuration in the config_path file and the
        `ResultData` in the experiment_info_['results_path'] pickle file.
        If the files already exist, they will be overwritten.
        """

        config_path = Path(self.base_path) / self.config_path
        results_path = Path(self.base_path) / self.experiment_info_["results_path"]

        # Set time stamps
        if "created_at" in self.experiment_info_:
            self.experiment_info_["updated_at"] = time.time()
        else:
            self.experiment_info_["created_at"] = time.time()
            self.experiment_info_["updated_at"] = time.time()

        with open(config_path, "w") as f:
            json.dump(self.experiment_info_, f, indent=4)

        with open(results_path, "wb") as f:
            pickle.dump(self.result_, f)

    def delete(self, missing_ok=False):
        """Deletes the experiment configuration file (json) and the ResultData file
        (pickle) from the disk.

        Parameters
        ----------
        missing_ok: bool, optional, default=False
            If True, the method will not raise an error if the files do not exist.

        Raises
        ------
        FileNotFoundError
            If the experiment configuration file or the `ResultData` file does not exist
            and `missing_ok` is False.

        Returns
        -------
        bool
            True if the files were deleted successfully.
        """

        config_path = Path(self.base_path) / self.config_path
        results_path = Path(self.base_path) / self.experiment_info_["results_path"]

        config_path.unlink(missing_ok=missing_ok)
        results_path.unlink(missing_ok=missing_ok)

        return True
