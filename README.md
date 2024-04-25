# REMAYN: REsults MAde easY in pythoN

`remayn` is an open-source Python toolkit focused on results management for machine learning experiments.
It includes the required functionalities to save the complete results of an experiment, load them, and generate repots.

| Overview  |                                                                                                                                          |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD** | [![!python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)](https://www.python.org/) |
| **Code**  | [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Linter: Ruff](https://img.shields.io/badge/Linter-Ruff-brightgreen?style=flat-square)](https://github.com/charliermarsh/ruff)                     |

## ⚙️ Installation

`remayn v0.1.0` is the last version supported by Python >=3.8.

The easiest way to install `remayn` is via `pip`, from this main branch of this GitHub repository:

    pip install git+https://github.com/ayrna/remayn.git@main


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
