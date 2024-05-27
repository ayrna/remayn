import pickle

import numpy as np
import pytest

from remayn.result import ResultData


def ExampleModel():
    def __eq__(self, other):
        return True


def random_targets():
    return np.random.randint(0, 10, 500)


def random_predictions():
    return np.random.rand(500, 10)


def random_history():
    return np.random.rand(100)


@pytest.fixture
def result_data():
    return ResultData(
        targets=random_targets(),
        predictions=random_predictions(),
        train_targets=random_targets(),
        train_predictions=random_predictions(),
        val_targets=random_targets(),
        val_predictions=random_predictions(),
        time=86754.5,
        train_history=random_history(),
        val_history=random_history(),
        best_params={"lr": 1e-3, "bs": 256, "alpha": 0.5, "beta": 1.2},
        best_model=ExampleModel(),
    )


def test_pickle_unpickle(result_data, tmp_path):
    pkl_path = tmp_path / "result_data.pkl"

    # Try to pickle and unpickle the data
    with open(pkl_path, "wb") as f:
        pickle.dump(result_data, f)

    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Check if the data is the same
    assert np.all(data.targets == result_data.targets)
    assert np.all(data.predictions == result_data.predictions)
    assert np.all(data.train_targets == result_data.train_targets)
    assert np.all(data.train_predictions == result_data.train_predictions)
    assert np.all(data.val_targets == result_data.val_targets)
    assert np.all(data.val_predictions == result_data.val_predictions)
    assert data.time == result_data.time
    assert np.all(data.train_history == result_data.train_history)
    assert np.all(data.val_history == result_data.val_history)
    assert data.best_params == result_data.best_params
    assert data.best_model == result_data.best_model
    assert data == result_data

    data.predictions = random_predictions()
    assert data != result_data


def test_validation():
    with pytest.raises(TypeError):
        ResultData()

    with pytest.raises(TypeError):
        ResultData(targets=random_targets())

    with pytest.raises(TypeError):
        ResultData(predictions=random_predictions())

    with pytest.raises(TypeError):
        ResultData(targets=None, predictions=None)

    ResultData(targets=[], predictions=[])
    ResultData(targets=random_targets(), predictions=[])
    ResultData(targets=[], predictions=random_predictions())
    ResultData(
        targets=random_targets(),
        predictions=random_predictions(),
        train_targets=[],
    )
    ResultData(
        targets=random_targets(),
        predictions=random_predictions(),
        train_predictions=[],
    )

    ResultData(
        targets=random_targets(),
        predictions=random_predictions(),
        val_targets=[],
    )
    ResultData(
        targets=random_targets(),
        predictions=random_predictions(),
        val_predictions=[],
    )

    with pytest.raises(TypeError):
        ResultData(
            targets=random_targets(),
            predictions=random_predictions(),
            time="time",
        )

    ResultData(
        targets=random_targets(),
        predictions=random_predictions(),
        train_history=[],
    )
    ResultData(
        targets=random_targets(),
        predictions=random_predictions(),
        val_history=[],
    )

    with pytest.raises(TypeError):
        ResultData(
            targets=random_targets(),
            predictions=random_predictions(),
            best_params="best_params",
        )
