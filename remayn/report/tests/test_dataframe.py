import numpy as np
import pandas as pd
import pytest

from remayn.report import create_2d_report


@pytest.fixture
def df1():
    return pd.DataFrame(
        data={
            "estimator_name": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "dataset": ["X", "X", "Y", "Y", "X", "X", "Y", "Y"],
            "rs": [1, 2, 1, 2, 1, 2, 1, 2],
            "QWK": [0.1, 0.2, 0.15, 0.25, 0.3, 0.1, 0.2, 0.25],
        }
    )


def test_create_2d_report_mean(df1):
    result = create_2d_report(
        source_df=df1,
        row_name="estimator_name",
        column_name="dataset",
        value_name="QWK",
        reduction="mean",
        compute_means="both",
        compute_ranks="both",
        ascending_ranks=False,
    )

    expected = pd.DataFrame(
        data={
            "X": {
                "A": 0.1500,
                "B": 0.2000,
                "mean": 0.1750,
                "rank": 2.0000,
            },
            "Y": {
                "A": 0.2000,
                "B": 0.2250,
                "mean": 0.2125,
                "rank": 1.0000,
            },
            "mean": {
                "A": 0.1750,
                "B": 0.2125,
                "mean": 0.19375,
                "rank": np.nan,
            },
            "rank": {
                "A": 2.0,
                "B": 1.0,
                "mean": np.nan,
                "rank": np.nan,
            },
        },
    )
    expected.index.name = "estimator_name"
    expected.columns.name = "dataset"

    pd.testing.assert_frame_equal(result, expected, rtol=1e-4)


def test_create_2d_report_std(df1):
    result = create_2d_report(
        source_df=df1,
        row_name="estimator_name",
        column_name="dataset",
        value_name="QWK",
        reduction="std",
        compute_means="both",
        compute_ranks="both",
        ascending_ranks=True,
    )

    expected = pd.DataFrame(
        data={
            "X": {
                "A": 0.0707,
                "B": 0.1414,
                "mean": 0.1061,
                "rank": 1.7500,
            },
            "Y": {
                "A": 0.0707,
                "B": 0.0354,
                "mean": 0.0530,
                "rank": 1.2500,
            },
            "mean": {
                "A": 0.0707,
                "B": 0.0884,
                "mean": 0.0795,
                "rank": np.nan,
            },
            "rank": {
                "A": 1.5,
                "B": 1.5,
                "mean": np.nan,
                "rank": np.nan,
            },
        },
    )
    expected.index.name = "estimator_name"
    expected.columns.name = "dataset"

    print(result)
    print(expected)

    pd.testing.assert_frame_equal(result, expected, rtol=5e-3)
