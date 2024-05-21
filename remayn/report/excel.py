from pathlib import Path
from typing import List, Optional, Union

import pandas as pd


def create_excel_summary_report(
    df: pd.DataFrame,
    destination_path: Union[str, Path],
    group_columns: List[str],
    *,
    individual_sheet_name: str = "Individual",
    average_sheet_name: str = "Average",
    std_sheet_name: str = "Std",
    excel_writer: Optional[pd.ExcelWriter] = None,
):
    """Creates a summary report of the given DataFrame and saves it to an Excel file.
    It groups the rows using the given `group_columns` and calculates the mean and
    standard deviation of the numeric columns for each group.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to create the report from.
    destination_path : Union[str, Path]
        The path to save the Excel file. It is ignored if `excel_writer` is provided.
    group_columns : List[str]
        The columns to group the rows by.
    individual_sheet_name : str, optional, default="Individual"
        The name of the sheet for the individual rows.
    average_sheet_name : str, optional, default="Average"
        The name of the sheet for the mean values.
    std_sheet_name : str, optional, default="Std"
        The name of the sheet for the standard deviation values.
    excel_writer : Optional[pd.ExcelWriter], optional, default=None
        A pd.ExcelWriter object that will be used to write the dataframes to an excel
        file. If None, a new Excel file will be created, by default None. Using an
        external ExcelWriter can be useful if you want to write additional sheets to
        the report excel file.

    Returns
    -------
    Path
        The path to the created Excel file.
    """

    destination_path = Path(destination_path)

    mean_df = df.groupby(group_columns).mean(numeric_only=True).reset_index()
    std_df = df.groupby(group_columns).std(numeric_only=True).reset_index()

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    if excel_writer is None:
        with pd.ExcelWriter(destination_path) as writer:
            df.to_excel(writer, sheet_name=individual_sheet_name, index=False)
            mean_df.to_excel(writer, sheet_name=average_sheet_name, index=False)
            std_df.to_excel(writer, sheet_name=std_sheet_name, index=False)
    else:
        df.to_excel(excel_writer, sheet_name=individual_sheet_name, index=False)
        mean_df.to_excel(excel_writer, sheet_name=average_sheet_name, index=False)
        std_df.to_excel(excel_writer, sheet_name=std_sheet_name, index=False)

    return destination_path


def create_excel_columns_report(
    df: pd.DataFrame,
    destination_path: Union[str, Path],
    metric_columns: List[str],
    pivot_index: str,
    pivot_columns: List[str],
    *,
    excel_writer: Optional[pd.ExcelWriter] = None,
):
    """Create an Excel report with multiple sheets, each containing a pivot table of the
    given DataFrame. The pivot table is created by pivoting the DataFrame using the
    given `pivot_index` and `pivot_columns`. The values of the pivot table are the
    columns specified in `metric_columns`. Each sheet in the Excel file corresponds to
    a column in `metric_columns`.

    To create an Excel file that contains one sheet for each metric, one column for each
    combination of methodology and dataset, and one row for each seed, the metric_columns
    should be the list of metric columns, the pivot_index should be the seed column and
    the pivot_columns should be the methodology and dataset columns.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to create the report from.
    destination_path : Union[str, Path]
        The path to save the Excel file. It is ignored if `excel_writer` is provided.
    metric_columns : List[str]
        The columns to create pivot tables for.
    pivot_index : str
        The column to use as the index of the pivot table.
    pivot_columns : List[str]
        The columns to use as the columns of the pivot table.
    excel_writer : Optional[pd.ExcelWriter], optional, default=None
        A pd.ExcelWriter object that will be used to write the dataframes to an excel
        file. If None, a new Excel file will be created, by default None. Using an
        external ExcelWriter can be useful if you want to write additional sheets to
        the report excel file.

    Returns
    -------
    Path
        The path to the created Excel file.
    """

    destination_path = Path(destination_path)

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_pivot_df(column):
        pivot_df = (
            df.pivot(
                index=pivot_index,
                columns=pivot_columns,
                values=column,
            )
            .reset_index()
            .drop(columns=pivot_index, level=0)
        )
        pivot_df.columns = pivot_df.columns.map("_".join)

        return pivot_df

    if excel_writer is None:
        with pd.ExcelWriter(destination_path) as writer:
            for column in metric_columns:
                pivot_df = _create_pivot_df(column)

                pivot_df.to_excel(writer, sheet_name=column, index=False)
    else:
        for column in metric_columns:
            pivot_df = _create_pivot_df(column)

            pivot_df.to_excel(excel_writer, sheet_name=column, index=False)

    return destination_path
