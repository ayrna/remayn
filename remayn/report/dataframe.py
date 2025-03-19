from typing import Literal, Optional, Union

import pandas as pd


def create_2d_report(
    source_df: pd.DataFrame,
    row_name: str = "row",
    column_name: str = "column",
    value_name: Optional[str] = None,
    reduction: Literal["mean", "std", "max", "min"] = "mean",
    compute_means: Union[None, Literal["row", "column", "both"]] = "row",
    compute_ranks: Union[None, Literal["row", "column", "both"]] = "row",
    ascending_ranks: bool = False,
):
    """Creates a dataframe with 2 dimensions from a source dataframe. The `row_name` and
    `column_name` indicate the columns from the source dataframe that will be used to
    create the rows and columns of the new dataframe. The values of the new dataframe
    will be the results of the `reduction` method applied to the values of the source
    dataframe that have the same value in the `row_name` and `column_name` columns.

    Parameters
    ----------
    source_df : pd.DataFrame
        Dataframe with all the individual results.

    row_name : str, default = 'method'
        Name of the column in the `source_df` that will be used to create the rows.
        The number of rows will be the number of unique values in this column.

    column_name : str, default = 'dataset'
        Name of the column in the `source_df` that will be used to create the columns.
        The number of columns will be the number of unique values in this column.

    value_name : str, default = None
        Name of the column in the `source_df` that will be used to create the values of
        the new dataframe. If `value_name` is None, the input dataframe can only contain
        one extra column apart from the `row_name` and `column_name`.

    reduction : str, default = 'mean'
        Method to reduce the results for different executions with the same value in
        the `row_name` and `column_name` columns. Possible values are `mean`, `std`,
        `max` or `min`.

    compute_means : str, default = 'row'
        Whether the mean of each row, column or both should be computed. Possible
        values are `row`, `column` or `both`.

    compute_ranks : str, default = 'row'
        Whether the rank of each row, column or both should be computed. Possible
        values are `row`, `column` or `both`.

    ascending_ranks : bool, default = False
        Whether the ranks should be computed in ascending order. If False, the ranks
        will be computed in descending order.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with the results aggregated by the `reduction` method.
    """

    if compute_means not in [None, "row", "column", "both"]:
        raise ValueError(
            "The compute_means parameter can only be None, 'row', 'column' or 'both'."
        )

    if compute_ranks not in [None, "row", "column", "both"]:
        raise ValueError(
            "The compute_ranks parameter can only be None, 'row', 'column' or 'both'."
        )

    df = source_df.groupby([row_name, column_name])

    if reduction == "mean":
        df = df.mean(numeric_only=True)
    elif reduction == "std":
        df = df.std(numeric_only=True)
    elif reduction == "max":
        df = df.max(numeric_only=True)
    elif reduction == "min":
        df = df.min(numeric_only=True)
    else:
        raise ValueError(f"Reduction method {reduction} not supported")

    df = df.reset_index().sort_values([row_name, column_name])

    # Check that there is only one more column apart from the row_name and column_name
    if len(df.columns) != 3 and value_name is None:
        print(df)
        raise ValueError(
            "If `value_name` is None, the input dataframe can only contain one extra"
            "column apart from the `row_name` and `column_name`."
        )

    if value_name is None:
        value_name = df.drop(columns=[row_name, column_name]).columns[0]

    df = df.pivot(index=row_name, columns=column_name, values=value_name)

    df_without_mean = df.copy()
    if compute_means == "row":
        df["mean"] = df.mean(axis=1)
    elif compute_means == "column":
        df.loc["mean"] = df.mean()
    elif compute_means == "both":
        df.loc["mean"] = df.mean()
        df["mean"] = df.mean(axis=1)

    if compute_ranks == "row":
        ranked_df = df_without_mean.rank(
            method="average", axis=0, ascending=ascending_ranks
        )
        df["rank"] = ranked_df.mean(axis=1)
    elif compute_ranks == "column":
        ranked_df = df_without_mean.rank(
            method="average", axis=1, ascending=ascending_ranks
        )
        df.loc["rank"] = ranked_df.mean()
    elif compute_ranks == "both":
        ranked_df = df_without_mean.rank(
            method="average", axis=0, ascending=ascending_ranks
        )
        df["rank"] = ranked_df.mean(axis=1)
        ranked_df = df_without_mean.rank(
            method="average", axis=1, ascending=ascending_ranks
        )
        df.loc["rank"] = ranked_df.mean()

    return df
