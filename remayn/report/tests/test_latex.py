import numpy as np
import pandas as pd
import pytest

from remayn.report import create_latex_2d_2values_report


@pytest.fixture
def mean_df():
    df = pd.DataFrame(
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
                "mean": 0.2126,
                "rank": 1.0000,
            },
            "mean": {
                "A": 0.1750,
                "B": 0.2126,
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
    df.index.name = "estimator_name"
    df.columns.name = "dataset"
    return df


@pytest.fixture
def std_df():
    df = pd.DataFrame(
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
    df.index.name = "estimator_name"
    df.columns.name = "dataset"
    return df


def test_create_latex_2d_2values_report(mean_df, std_df):
    result = create_latex_2d_2values_report(mean_df, std_df)

    df = pd.DataFrame(
        data={
            "X": {
                "A": "$0.150_{0.071}$",
                "B": "$0.200_{0.141}$",
                "mean": "$0.175_{0.106}$",
                "rank": "$2.000_{1.750}$",
            },
            "Y": {
                "A": "$0.200_{0.071}$",
                "B": "$0.225_{0.035}$",
                "mean": "$0.213_{0.053}$",
                "rank": "$1.000_{1.250}$",
            },
            "mean": {
                "A": "$0.175_{0.071}$",
                "B": "$0.213_{0.088}$",
                "mean": "$0.194_{0.080}$",
                "rank": "",
            },
            "rank": {
                "A": "$2.000_{1.500}$",
                "B": "$1.000_{1.500}$",
                "mean": "",
                "rank": "",
            },
        },
    )
    df.columns.name = "dataset"
    df.index.name = "estimator_name"

    print(result)
    print(df)

    pd.testing.assert_frame_equal(result, df)


def test_create_latex_2d_2values_report_custom_float(mean_df, std_df):
    result = create_latex_2d_2values_report(
        mean_df,
        std_df,
        float_formatter=lambda x: f"{x:.2f}",
    )

    df = pd.DataFrame(
        data={
            "X": {
                "A": "$0.15_{0.07}$",
                "B": "$0.20_{0.14}$",
                "mean": "$0.17_{0.11}$",
                "rank": "$2.00_{1.75}$",
            },
            "Y": {
                "A": "$0.20_{0.07}$",
                "B": "$0.23_{0.04}$",
                "mean": "$0.21_{0.05}$",
                "rank": "$1.00_{1.25}$",
            },
            "mean": {
                "A": "$0.17_{0.07}$",
                "B": "$0.21_{0.09}$",
                "mean": "$0.19_{0.08}$",
                "rank": "",
            },
            "rank": {
                "A": "$2.00_{1.50}$",
                "B": "$1.00_{1.50}$",
                "mean": "",
                "rank": "",
            },
        },
    )
    df.columns.name = "dataset"
    df.index.name = "estimator_name"

    print(result)
    print(df)

    pd.testing.assert_frame_equal(result, df)
