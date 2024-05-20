import pandas as pd
import pandas.testing as pdt
import pytest

from remayn.report import create_excel_columns_report, create_excel_summary_report


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "estimator": [
                "A",
                "A",
                "B",
                "B",
                "A",
                "A",
                "B",
                "B",
                "A",
                "A",
                "B",
                "B",
                "A",
                "A",
                "B",
                "B",
            ],
            "dataset": [
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
                "X",
                "Y",
            ],
            "seed": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
            "accuracy": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.15,
                0.25,
                0.35,
                0.45,
                0.2,
                0.3,
                0.4,
                0.5,
                0.25,
                0.35,
                0.45,
                0.55,
            ],
            "precision": [
                0.5,
                0.6,
                0.7,
                0.8,
                0.55,
                0.65,
                0.75,
                0.85,
                0.6,
                0.7,
                0.8,
                0.9,
                0.65,
                0.75,
                0.85,
                0.95,
            ],
        }
    )


@pytest.fixture
def excel_path(tmp_path):
    return tmp_path / "test.xlsx"


def test_create_excel_summary_report(df, excel_path):
    group_columns = ["estimator", "dataset"]
    ret_v = create_excel_summary_report(df, excel_path, group_columns)

    assert excel_path.exists()
    assert pd.read_excel(excel_path, sheet_name="Individual").equals(df)
    average_df = pd.read_excel(excel_path, sheet_name="Average")
    for column in [c for c in df.columns if c not in group_columns]:
        assert column in average_df.columns

    std_df = pd.read_excel(excel_path, sheet_name="Std")
    for column in [c for c in df.columns if c not in group_columns]:
        assert column in std_df.columns

    pdt.assert_frame_equal(
        average_df,
        df.groupby(group_columns).mean(numeric_only=True).reset_index(),
    )

    pdt.assert_frame_equal(
        std_df,
        df.groupby(group_columns).std(numeric_only=True).reset_index(),
    )

    assert ret_v == excel_path


def test_create_excel_summary_report_custom_sheet_names(df, excel_path):
    group_columns = ["estimator", "dataset"]
    individual_sheet_name = "Original"
    average_sheet_name = "Mean"
    std_sheet_name = "Standard Deviation"
    ret_v = create_excel_summary_report(
        df,
        excel_path,
        group_columns,
        individual_sheet_name=individual_sheet_name,
        average_sheet_name=average_sheet_name,
        std_sheet_name=std_sheet_name,
    )

    assert excel_path.exists()
    assert pd.read_excel(excel_path, sheet_name=individual_sheet_name).equals(df)
    average_df = pd.read_excel(excel_path, sheet_name=average_sheet_name)
    for column in [c for c in df.columns if c not in group_columns]:
        assert column in average_df.columns

    std_df = pd.read_excel(excel_path, sheet_name=std_sheet_name)
    for column in [c for c in df.columns if c not in group_columns]:
        assert column in std_df.columns

    pdt.assert_frame_equal(
        average_df,
        df.groupby(group_columns).mean(numeric_only=True).reset_index(),
    )

    pdt.assert_frame_equal(
        std_df,
        df.groupby(group_columns).std(numeric_only=True).reset_index(),
    )

    assert ret_v == excel_path


def test_create_excel_summary_report_shared_writer(df, excel_path):
    group_columns = ["estimator", "dataset"]
    with pd.ExcelWriter(excel_path) as writer:
        ret_v = create_excel_summary_report(
            df, excel_path, group_columns, excel_writer=writer
        )
        ret_v2 = create_excel_summary_report(
            df,
            excel_path,
            group_columns,
            individual_sheet_name="Original",
            average_sheet_name="Mean",
            std_sheet_name="Standard Deviation",
            excel_writer=writer,
        )

    assert excel_path.exists()
    assert pd.read_excel(excel_path, sheet_name="Individual").equals(df)
    assert pd.read_excel(excel_path, sheet_name="Original").equals(df)
    average_df = pd.read_excel(excel_path, sheet_name="Average")
    for column in [c for c in df.columns if c not in group_columns]:
        assert column in average_df.columns

    mean_df = pd.read_excel(excel_path, sheet_name="Mean")
    for column in [c for c in df.columns if c not in group_columns]:
        assert column in mean_df.columns

    std_df = pd.read_excel(excel_path, sheet_name="Std")
    for column in [c for c in df.columns if c not in group_columns]:
        assert column in std_df.columns

    std2_df = pd.read_excel(excel_path, sheet_name="Standard Deviation")
    for column in [c for c in df.columns if c not in group_columns]:
        assert column in std2_df.columns

    pdt.assert_frame_equal(
        average_df,
        df.groupby(group_columns).mean(numeric_only=True).reset_index(),
    )

    pdt.assert_frame_equal(
        mean_df,
        df.groupby(group_columns).mean(numeric_only=True).reset_index(),
    )

    pdt.assert_frame_equal(
        std_df,
        df.groupby(group_columns).std(numeric_only=True).reset_index(),
    )

    pdt.assert_frame_equal(
        std2_df,
        df.groupby(group_columns).std(numeric_only=True).reset_index(),
    )

    assert ret_v == excel_path
    assert ret_v2 == excel_path


def test_create_excel_columns_report(df, excel_path):
    metric_columns = ["accuracy", "precision"]
    pivot_index = "seed"
    pivot_columns = ["estimator", "dataset"]
    ret_v = create_excel_columns_report(
        df, excel_path, metric_columns, pivot_index, pivot_columns
    )

    assert excel_path.exists()
    for column in metric_columns:
        pivot_df = (
            df.pivot(
                index=pivot_index,
                columns=pivot_columns,
                values=column,
            )
            .reset_index()
            .drop(columns=pivot_index, level=0)
        )
        read_df = pd.read_excel(excel_path, sheet_name=column)
        assert read_df.values == pytest.approx(pivot_df.values)

        assert len(read_df.columns) == df[pivot_columns].drop_duplicates().shape[0]
        assert len(read_df) == df[pivot_index].nunique()

    assert ret_v == excel_path


def test_create_excel_columns_report_shared_writer(df, excel_path):
    metric_columns = ["accuracy", "precision"]
    pivot_index = "seed"
    pivot_columns = ["estimator", "dataset"]
    with pd.ExcelWriter(excel_path) as writer:
        ret_v = create_excel_columns_report(
            df,
            excel_path,
            metric_columns,
            pivot_index,
            pivot_columns,
            excel_writer=writer,
        )
        ret_v2 = create_excel_columns_report(
            df,
            excel_path,
            metric_columns,
            pivot_index,
            pivot_columns,
            excel_writer=writer,
        )

    assert excel_path.exists()
    for column in metric_columns:
        pivot_df = (
            df.pivot(
                index=pivot_index,
                columns=pivot_columns,
                values=column,
            )
            .reset_index()
            .drop(columns=pivot_index, level=0)
        )
        read_df = pd.read_excel(excel_path, sheet_name=column)
        assert read_df.values == pytest.approx(pivot_df.values)

        assert len(read_df.columns) == df[pivot_columns].drop_duplicates().shape[0]
        assert len(read_df) == df[pivot_index].nunique()

    assert ret_v == excel_path
    assert ret_v2 == excel_path
