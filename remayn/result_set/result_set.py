import importlib.util
import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..result import Result
from ..utils import sanitize_json
from .utils import get_row_from_result


class ResultSet:
    base_path: Path
    results: list[Result]

    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.results = []

        self.load()

    def __str__(self):
        return (
            "ResultSet with"
            f" {len(self.results)} result{'s' if len(self.results) > 1 else ''}"
        )

    def __repr__(self):
        return self.__str__()

    def load(self):
        json_files = list(self.base_path.rglob("*.json"))
        pkl_files = list(self.base_path.rglob("*.pkl"))

        if len(json_files) != len(pkl_files):
            raise ValueError(
                f"Number of json files ({len(json_files)}) does not match"
                f" number of pkl files ({len(pkl_files)})",
            )

        for path in json_files:
            try:
                with open(path, "r") as f:
                    experiment_info = json.load(f)
            except json.JSONDecodeError:
                warnings.warn(
                    f"Could not load json file {path}. Skipping this file.",
                    RuntimeWarning,
                )
            except FileNotFoundError:
                warnings.warn(
                    f"Could not find json file {path}. Skipping this file.",
                    RuntimeWarning,
                )

            result = Result(self.base_path, experiment_info)
            self.results.append(result)

    def filter(self, config={}):
        """Filters the results by config.

        Parameters
        ----------
        config : dict, optional, default={}
            The config fields to filter by. To add a result to the filtered set, the
            result's config must contain all the fields in the config parameter with the
            same values. For example, if config={"a": 1, "b": 2}, the result config must
            contain both fields with those values. However, the result config can contain
            additional fields not listed in the provided config dictionary.
        Returns
        -------
        results : ResultSet
            The filtered results.
        """

        safe_config = sanitize_json(config)
        config_json = json.dumps(safe_config, indent=4)
        config_from_json = json.loads(config_json)

        filtered_results = []
        for result in self.results:
            config = result.get_config()
            matches = True
            for k, v in config_from_json.items():
                if k not in config.keys() or config[k] != v:
                    matches = False
                    break

            if matches:
                filtered_results.append(result)

        return CustomResultSet(filtered_results)

    def create_dataframe(
        self,
        config_columns: list[str] = [],
        filter_fn: Callable[[Result], bool] = lambda result: True,
        metrics_fn: Callable[
            [np.ndarray, np.ndarray], dict[str, float]
        ] = lambda targets, predictions: {},
        include_train: bool = False,
        include_val: bool = False,
        best_params_columns: list[str] = [],
        n_jobs: int = -1,
    ):
        """Creates a pandas DataFrame that contains all the results stored in this
        ResultSet. The DataFrame will contain the columns specified in config_columns,
        best_params_columns, and the metrics computed by metrics_fn. The metrics will be
        computed on the test set by default, but the train and validation metrics can be
        included using the include_train and include_val flags. If filter_fn parameter
        is provided, only the results that satisfy the condition will be included in the
        DataFrame.

        Parameters
        ----------
        config_columns : list of str, optional, default=[]
            List of columns from the config to include in the dataframe.
        filter_fn : Callable[[ResultsFile], bool], optional, default=lambda result: True
            Function to filter the results to include in the dataframe. If it returns
            True, the result row will be included. The function receives a single
            parameter which is the Result object being processed. It must return a
            boolean value.
        metrics_fn : Callable[[np.ndarray, np.ndarray], dict[str, float]]
            Function that computes the metrics from the targets and predictions.
            It receives two numpy arrays, the targets and the predictions, and returns a
            dictionary where the key is the name of the metric and the value is the value
            of the metric. The shape of the numpy arrays depend on the kind of data that
            is stored within the ResultFile. While any shape can be valid, the
            implementation of the metrics function must be coherent with the data stored
            in the ResultFile.
        include_train : bool, optional, default=False
            Whether to include the metrics computed on the train set.
        include_val : bool, optional, default=False
            Whether to include the metrics computed on the validation set.
        best_params_columns : list of str, optional, default=[]
            List of columns from the best_params list to include in the dataframe.
        n_jobs : int, optional, default=-1
            The number of jobs to run in parallel (must be > 0). If -1, all CPUs are used.
            If 1 is given, no parallel computing code is used at all, which is useful for
            debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for
            n_jobs = -2, all CPUs but one are used.
            `joblib` is used for parallel processing. If it is not installed, n_jobs
            will be set to 1 and a warning will be issued.
            The parallel backend is not specified within this function, so the user
            can set it using the `joblib` API. Example:
            ```
            from joblib import parallel_backend
            with parallel_backend("loky", n_jobs=2):
                results = rs.get_dataframe(...)
            ```

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
                )
                for result in self.results
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
            for result in self.results:
                if filter_fn(result):
                    data.append(
                        get_row_from_result(
                            result,
                            config_columns,
                            metrics_fn,
                            include_train,
                            include_val,
                            best_params_columns,
                        )
                    )

        # Save the columns of the row with the highest number of columns to avoid columns
        # sorting issues
        for row in data:
            if row:
                if len(row.keys()) > len(columns):
                    columns = row.keys()

        return pd.DataFrame(data)

    def get_experiment(self, idx) -> Result:
        """Obtains the experiment at the given index.

        Parameters
        ----------
        idx : int
            The index of the experiment to obtain.

        Returns
        -------
        result : ResultsFile or None
            The experiment at the given index. If the index is out of bounds, returns None.

        """

        if idx < 0 or idx >= len(self.results):
            raise IndexError(f"Index {idx} out of bounds")
        return self.results[idx]

    def find_experiment(self, config, deep=False):
        """Find the first experiment with the given config.

        Parameters
        ----------
        config : dict
            The config to search for.

        deep : bool, optional, default=False
            Whether to use deep comparison of configs. Requires the deepdiff package.
            Recommended when using nested dictionaries or nan values.

        Returns
        -------
        result : ResultsFile or None
            The first result with the given config, or None if not found.

        """

        safe_config = sanitize_json(config)
        config_json = json.dumps(safe_config, indent=4)
        config_from_json = json.loads(config_json)

        if deep:
            try:
                from deepdiff import DeepDiff

                for result in self.results:
                    diff = DeepDiff(
                        result.get_config(),
                        config_from_json,
                        ignore_nan_inequality=True,
                        ignore_order=True,
                        ignore_numeric_type_changes=True,
                        significant_digits=6,
                        number_to_string_func=lambda x,
                        significant_digits,
                        number_format_notation: str(round(x, significant_digits)),
                    )
                    if diff == {}:
                        return result
            except ImportError:
                raise ImportError(
                    "deepdiff is required to use deep comparison of configs"
                )
        else:
            for result in self.results:
                if result.get_config() == config_from_json:
                    return result
        return None

    def __iter__(self):
        return iter(self.results)

    def __len__(self):
        return len(self.results)

    def __getitem__(self, idx):
        return self.results[idx]


class CustomResultSet(ResultSet):
    def __init__(self, results):
        self.results = results

    def load(self):
        raise NotImplementedError("CustomResultSet does not support load")
