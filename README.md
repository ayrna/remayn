# REMAYN: REsults MAde easY in pythoN

`remayn` is an open-source Python toolkit focused on results management for machine learning experiments.
It includes the required functionalities to save the complete results of an experiment, load them, and generate reports.

| Overview  |                                                                                                                                          |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD** | [![!codecov](https://img.shields.io/codecov/c/github/ayrna/remayn?label=codecov&logo=codecov)](https://codecov.io/gh/ayrna/remayn) [![!docs](https://readthedocs.org/projects/remayn/badge/?version=latest&style=flat)](https://remayn.readthedocs.io/en/latest/)  |
| **Code**  | [![!python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Linter: Ruff](https://img.shields.io/badge/Linter-Ruff-brightgreen?style=flat-square)](https://github.com/charliermarsh/ruff)                     |

## Getting started

### ⚙️ Installation

`remayn` is supported by Python >=3.8.

The easiest way to install `remayn` is via `pip`:

    pip install remayn

### 💾 Saving the results of a experiment
A new `Result` object can be created using the `make_result` function. Then, the `Result` can be saved to disk by simply calling the `save()` method.
```python
import numpy as np
from remayn.result import make_result

targets = np.array([1, 2, 3])
predictions = np.array([1.1, 2.2, 3.3])
config = {"model": "linear_regression", "dataset": "iris", "learning_rate": 1e-3}

result = make_result("./results",
                    config=config,
                    targets=targets,
                    predictions=predictions
                    )
result.save()
```
This will generate an unique identifier for this `Result` and it will be saved in a subdirectory of the `./results` directory.

### ⌛ Loading a set of results
After saving the results of all the experiments, the set of results can be loaded using the `ResultFolder` class, as shown in the following snippet:

```python
from remayn.result_set import ResultFolder

rs = ResultFolder('./results')
```
Note that the same path used to save the results is employed here to load the `ResultFolder`. The `ResultFolder` object is a special type of `ResultSet` and represents a set of results which have been loaded from disk.

### 📝 Creating a pandas DataFrame that contains all the results
After loading the results, the `create_dataframe` method of the `ResultSet` class can be used to generate a `pandas.DataFrame` containing all the results. This method receives a callable which is used to compute the metrics from the targets and predictions stored in each `Result`. Therefore, first we can define a function that computes the metrics:
```python
def mse(y_true, y_pred):
    return ((y_true - y_pred)**2).mean()

def _compute_metrics(targets, predictions):
    return {
        "mse": mse(targets, predictions),
    }
```

Then, the `create_dataframe` method of the `ResultSet` is used:

```python
from remayn.result_set import ResultFolder

rs = ResultFolder('./results')
df = rs.create_dataframe(
    config_columns=[
        "model",
        "dataset",
        "learning_rate",
    ],
    metrics_fn=_compute_metrics,
)
```

Finally, the DataFrame can be saved to a file by using the existing `pandas` methods:

```python
df.to_excel('results.xlsx', index=False)
```

This will generate an Excel file that contains the column given in the `config_columns` parameter along with the columns associated with the metrics computed in the function provided.

## Collaborating

Code contributions to the `remayn` project are welcomed via pull requests.
Please, contact the maintainers (maybe opening an issue) before doing any work to make sure that your contributions align with the project.

### Guidelines for code contributions

* You can clone the repository and then install the library from the local repository folder:

```bash
git clone git@github.com:ayrna/remayn.git
pip install ./remayn
```

* In order to set up the environment for development, install the project in editable mode and include the optional dev requirements:
```bash
pip install -e '.[dev]'
```
* Install the pre-commit hooks before starting to make any modifications:
```bash
pre-commit install
```
* Write code that is compatible with all supported versions of Python listed in the `pyproject.toml` file.
* Create tests that cover the common cases and the corner cases of the code.
* Preserve backwards-compatibility whenever possible, and make clear if something must change.
* Document any portions of the code that might be less clear to others, especially to new developers.
* Write API documentation as docstrings.
