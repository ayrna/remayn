import importlib.util
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Set, Union

import numpy as np
import pandas as pd

from ..result import Result
from ..utils import dict_contains_dict, sanitize_json
from .utils import get_row_from_result


class ResultSet:
    """Stores a set of `Result` objects.
    For each `Result` object, it stores the metadata of the experiment, such as the
    experiment info and the path where the result is stored. The predictions and targets
    are not loaded until needed.

    Attributes
    ----------
    results_ : Dict[str, Result]
        Dictionary that contains the config of the experiment as the key and the
        `Result` object as the value.
    """

    results_: Dict[str, Result]

    def __init__(self, results: Union[List[Result], Set[Result], Dict[str, Result]]):
        """Creates a `ResultSet` object.
        It can be initialised from a list, a set or a dictionary:
        - If a list or a set is provided, the config of the `Result` objects will be used
          as the key in the dictionary. If there are repeated configs, only the last
          `Result` object with that config will be stored.
        - If a dictionary is provided, it is assigned to the `results_` attribute.

        Parameters
        ----------
        results : Union[List[Result], Set[Result], Dict[str, Result]]
            The `Result` objects to store in the `ResultSet`.

        Raises
        ------
        TypeError
            If the `results` parameter is not a list, a set or a dictionary.
        """

        if isinstance(results, dict):
            self.results_ = results
        elif isinstance(results, (list, set)):
            self.results_ = {}
            for result in results:
                if not isinstance(result, Result):
                    raise TypeError(
                        f"Expected Result, got {type(result).__name__}",
                    )
                self.results_[str(sanitize_json(result.config))] = result
        else:
            raise TypeError(
                f"Expected list, set or dict, got {type(results).__name__}",
            )

    def contains(self, result: Union[str, dict, Result]) -> bool:
        """Checks if the `ResultSet` contains the given `Result`.

        Parameters
        ----------
        result : Union[str, dict, Result]
            The result that is searched in the `ResultSet`. It can be:
            - a string: the config dict of the result transformed to string,
            - a dict: the config dict of the result,
            - a `Result` object.

        If the result is a string, it is employed as the key to search in the
        `ResultSet`. If the result is a dict, it is json sanitized and transformed to
        string to be used as the key. If the result is a `Result` object, its config is
        sanitized and transformed to string to be used as the key.

        Returns
        -------
        contains : bool
            Whether the `ResultSet` contains the given result.

        Raises
        ------
        TypeError
            If the `result` parameter is not a str, a dict or a `Result`.
        """

        if isinstance(result, str):
            return result in self.results_
        elif isinstance(result, dict):
            return str(sanitize_json(result)) in self.results_
        elif isinstance(result, Result):
            return str(sanitize_json(result.config)) in self.results_
        else:
            raise TypeError(
                f"Expected str, dict or Result, got {type(result).__name__}",
            )

    def filter_by_config(self, config: dict) -> "ResultSet":
        """Filters the results by config.

        Parameters
        ----------
        config : dict
            The config fields to filter by. To add a result to the filtered set, the
            result's config must contain all the fields in the config parameter with the
            same values. For example, if config={"a": 1, "b": 2}, the result config must
            contain both fields with those values. However, the result config can contain
            additional fields not listed in the provided config dictionary.
        Returns
        -------
        results : ResultSet
            A `ResultSet` that contains only the results that match the given config.

        Raises
        ------
        TypeError
            If the `config` parameter is not a dict.
        """

        if not isinstance(config, dict):
            raise TypeError(
                f"Expected dict, got {type(config).__name__}",
            )

        safe_config = sanitize_json(config)
        # config_json = json.dumps(safe_config, indent=4)
        # config_from_json = json.loads(config_json)

        filtered_results = []
        for result in self:
            config = sanitize_json(result.config)
            if dict_contains_dict(config, safe_config):
                filtered_results.append(result)

        return ResultSet(filtered_results)

    def create_dataframe(
        self,
        config_columns: List[str] = [],
        filter_fn: Callable[[Result], bool] = lambda result: True,
        metrics_fn: Callable[
            [np.ndarray, np.ndarray], Dict[str, float]
        ] = lambda targets, predictions: {},
        include_train: bool = False,
        include_val: bool = False,
        best_params_columns: List[str] = [],
        n_jobs: int = -1,
        config_columns_prefix: str = "config_",
        best_params_columns_prefix: str = "best_",
    ):
        """Creates a pandas.DataFrame that contains all the results stored in this
        ResultSet. The DataFrame will contain the columns specified in config_columns,
        best_params_columns, and the metrics computed by metrics_fn. The metrics will be
        computed on the test set by default, but the train and validation metrics can be
        included using the include_train and include_val flags. If filter_fn parameter
        is provided, only the results that satisfy the condition will be included in the
        DataFrame.

        Parameters
        ----------
        config_columns : List[str], optional, default=[]
            List of columns from the config to include in the dataframe.
        filter_fn : Callable[[ResultData], bool], optional, default=lambda result: True
            Function to filter the results to include in the dataframe. If it returns
            True, the result row will be included. The function receives a single
            parameter which is the Result object being processed. It must return a
            boolean value.
        metrics_fn : Callable[[np.ndarray, np.ndarray], Dict[str, float]]
            Function that computes the metrics from the targets and predictions.
            It receives two numpy arrays, the targets and the predictions, and returns a
            dictionary where the key is the name of the metric and the value is the value
            of the metric. The shape of the numpy arrays depend on the kind of data that
            is stored within the ResultData. While any shape can be valid, the
            implementation of the metrics function must be coherent with the data stored
            in the ResultData.
        include_train : bool, optional, default=False
            Whether to include the metrics computed on the train set.
        include_val : bool, optional, default=False
            Whether to include the metrics computed on the validation set.
        best_params_columns : List[str], optional, default=[]
            List of columns from the best_params list to include in the dataframe.
        n_jobs : int, optional, default=-1
            The number of jobs to run in parallel (must be > 0). If -1, all CPUs are used.
            If 1 is given, no parallel computing code is used at all, which is useful for
            debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used.
            joblib is used for parallel processing. If it is not installed, n_jobs
            will be set to 1 and a warning will be issued.
            The parallel backend is not specified within this function, so the user
            can set it using the `parallel_backend` of the joblib API.
        config_columns_prefix : str, optional, default = "config\_"
            The prefix to add to the config columns. If '', no prefix is added. Note
            that using an empty prefix can result in column name conflicts.
        best_params_columns_prefix : str, optional, default = "best\_"
            The prefix to add to the best_params columns. If empty string, no prefix is added. Note
            that using an empty prefix can result in column name conflicts.

        Returns
        -------
        df : pandas dataframe
            The dataframe with the results.

        Raises
        ------
        ValueError
            If n_jobs is set to 0.
        """

        data = []
        columns = []

        if n_jobs == 0:
            raise ValueError(
                "n_jobs=0 is not supported. See n_jobs parameter in the documentation."
            )

        # Set n_jobs to 1 if joblib is not available
        if n_jobs != 1 and not importlib.util.find_spec("joblib"):
            warnings.warn(
                "joblib is not installed. n_jobs will be set to 1. To use parallel"
                " processing, install joblib. Set n_jobs=1 to run sequentially.",
                RuntimeWarning,
            )
            n_jobs = 1

        if n_jobs != 1:
            # Parallel processing
            import joblib

            # Populate the list of jobs that will be run in parallel
            joblist = [
                (
                    result,
                    config_columns,
                    metrics_fn,
                    include_train,
                    include_val,
                    best_params_columns,
                    config_columns_prefix,
                    best_params_columns_prefix,
                )
                for result in self
                if filter_fn(result)
            ]

            with joblib.Parallel(n_jobs=n_jobs) as parallel:
                data = (
                    list(
                        parallel(
                            joblib.delayed(get_row_from_result)(*args)
                            for args in joblist
                        )
                    )
                    or []
                )
        else:
            for result in self:
                if filter_fn(result):
                    data.append(
                        get_row_from_result(
                            result,
                            config_columns,
                            metrics_fn,
                            include_train,
                            include_val,
                            best_params_columns,
                            config_columns_prefix,
                            best_params_columns_prefix,
                        )
                    )

        # Save the columns of the row with the highest number of columns to avoid columns
        # sorting issues
        for row in data:
            if row:
                if len(row.keys()) > len(columns):
                    columns = row.keys()

        return pd.DataFrame(data)

    def add(self, result: Result):
        """Adds a `Result` object to the `ResultSet`. If the result already exists in
        the `ResultSet`, it will be replaced by the new one.

        Parameters
        ----------
        result : `Result`
            The `Result` object to add.

        Raises
        ------
        TypeError
            If the `result` parameter is not a `Result`.
        """

        if not isinstance(result, Result):
            raise TypeError(
                f"Expected Result, got {type(result).__name__}",
            )

        self.results_[str(sanitize_json(result.config))] = result

    def remove(self, key: Union[str, dict, Result]):
        """Removes a `Result` object identified by a key from the `ResultSet`.

        Parameters
        ----------
        key : Union[str, dict, Result]
            The key of the `Result` to remove. It can be given as:
            - a string: the config dict of the result transformed to string,
            - a dict: the config dict of the result,
            - a `Result` object.

        Raises
        ------
        TypeError
            If the `key` parameter is not a str, a dict or a `Result`.
        KeyError
            If the key is not found in the `ResultSet`.
        """

        if isinstance(key, str):
            del self.results_[key]
        elif isinstance(key, dict):
            del self.results_[str(sanitize_json(key))]
        elif isinstance(key, Result):
            del self.results_[str(sanitize_json(key.config))]
        else:
            raise TypeError(
                f"Expected str, dict or Result, got {type(key).__name__}",
            )

    def __iter__(self):
        """Returns an iterator to the `Result` objects contained in this `ResultSet`.

        Returns
        -------
        iter : Iterator[Result]
            An iterator to the `Result` objects in the `ResultSet`.
        """
        return iter(self.results_.values())

    def __len__(self):
        return len(self.results_)

    def __getitem__(self, key: Union[str, dict, Result]):
        """Gets an item from the `ResultSet` by its key.

        The key can be one of:
        - a string: the config dict of the result transformed to string,
        - a dict: the config dict of the result,
        - a `Result` object.

        If the key is a string, it is employed as the key to search in the
        `ResultSet`. If the key is a dict, it is json sanitized and transformed to
        string to be used as the key. If the key is a `Result` object, its config is
        sanitized and transformed to string to be used as the key.

        Parameters
        ----------
        key : Union[str, dict, Result]
            The key to search in the `ResultSet`.

        Returns
        -------
        result : `Result`
            The `Result` object that corresponds to the key.
        """

        if isinstance(key, str):
            return self.results_[key]
        elif isinstance(key, dict):
            return self.results_[str(sanitize_json(key))]
        elif isinstance(key, Result):
            return self.results_[str(sanitize_json(key.config))]
        else:
            raise TypeError(
                f"Expected str, dict or Result, got {type(key).__name__}",
            )

    def __str__(self):
        return (
            "ResultSet with"
            f" {len(self.results_)} result{'s' if len(self.results_) > 1 else ''}"
        )

    def __repr__(self):
        return self.__str__()

    def __contains__(self, result: Union[str, dict, Result]) -> bool:
        return self.contains(result)


class ResultFolder(ResultSet):
    """Stores a set of set of `Result` objects loaded from a directory.
    For each `Result` object, it stores the metadata of the experiment, such as the
    experiment info and the path where the result is stored. The predictions and targets
    are not loaded until needed.

    Attributes
    ----------
    base_path : Path
        The path where the results are stored.
    results_ : list of Result
        The list of `Result` objects stored in the `ResultSet`.
    """

    base_path: Path
    results_: Dict[str, Result]

    def __init__(self, base_path):
        self.base_path = Path(base_path)

        super().__init__(results=[])

        self.load()

    def load(self):
        """Loads the experiment info of all the results from the `base_path` directory.
        Only the metadata of the experiments is loaded, while the `ResultData` is not
        loaded until needed.

        It retrieves all the json files from `base_path`. Also, it checks that each
        json file has a corresponding pkl file. If the number of json files does not
        match the number of pkl files, a ValueError is raised.

        Raises
        ------
        ValueError
            If the the pickle file for a given json file is not found.

        Examples
        --------
        >>> from remayn.result_set import ResultFolder
        >>> rf = ResultFolder("./results")
        """

        self.results_ = {}

        json_files = list(self.base_path.rglob("*.json"))
        pkl_files = list(self.base_path.rglob("*.pkl"))

        # Check that all the json files have its corresponding pkl file
        if len(json_files) != len(pkl_files):
            for json_file in json_files:
                if json_file.with_suffix(".pkl") not in pkl_files:
                    raise FileNotFoundError(
                        f"Could not find pkl file for json file {json_file}",
                    )
            warnings.warn(
                f"Number of json files ({len(json_files)}) does not match"
                f" number of pkl files ({len(pkl_files)})",
            )

        for path in json_files:
            result = Result.load(self.base_path, path.stem)
            key = str(sanitize_json(result.config))
            self.results_[key] = result
