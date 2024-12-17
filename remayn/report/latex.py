from typing import Callable, Literal

import numpy as np
import pandas as pd


def create_latex_2d_2values_report(
    df_l1: pd.DataFrame,
    df_l2: pd.DataFrame,
    float_formatter: Callable[[float], str] = lambda x: f"{x:.3f}",
    highlight: dict[str, Literal[None, "max", "min"]] = {},
    highlight_excluded: list[str] = [],
    highlight_axis: int = 0,
):
    """Creates a LaTeX table from two dataframes with the same shape. The first
    dataframe will be used to create the values of the cells and the second dataframe
    will be used to create the subscripts of the cells. The `highlight` parameter can be
    used to highlight the best value in each row or column.

    Parameters
    ----------
    df_l1 : pd.DataFrame
        Level 1 dataframe. This dataframe will be used to create the values of the cells.

    df_l2 : pd.DataFrame
        Level 2 dataframe. This dataframe will be used to create the subscripts of the cells.

    float_formatter : Callable[[float], str], default = lambda x: f"{x:.3f}"
        Function to format the float values.

    highlight : dict[str, Literal[None, "max", "min"]], default = {}
        Dictionary with the highlight mode for each row or column. The key "*" will be
        used for the default highlight mode. The possible values are `None`, `max` or
        `min`. For example: {"*": "max", "rank": "min"} will highlight the maximum
        value in all columns except for the "rank" column where the minimum value will be
        highlighted.

    highlight_excluded : list[str], default = []
        List of rows/columns that will not be considered for the highlight. If
        `highlight_axis` is 0, rows whose index match the elements of this list will
        be excluded. If `highlight_axis` is 1, columns whose name match the elements
        of this list will be excluded. Common values are "mean" or "rank".

    highlight_axis : int, default = 1
        Axis to apply the highlight. If 0, the best value in each column will be
        highlighted. If 1, the best value in each row will be highlighted.

    Returns
    -------
    df: pd.DataFrame
        Dataframe with LaTex format where each cell has the value of the `df_l1` and the
        subscript of the `df_l2`. If `highlight` is used, the best value and the second
        best value will be highlighted in bold and italic, respectively.
    """

    def highlight_best(s):
        hl_mode = highlight.get(s.name, highlight.get("*", None))

        # Drop elements that are in the highlight_excluded list
        s_excluded = s.drop(highlight_excluded)

        if hl_mode == "max":
            best_idx = s_excluded.idxmax()
            second_best_idx = s_excluded.drop(best_idx).idxmax()
        elif hl_mode == "min":
            best_idx = s_excluded.idxmin()
            second_best_idx = s_excluded.drop(best_idx).idxmin()
        else:
            best_idx = None
            second_best_idx = None

        s = s.apply(float_formatter)
        if best_idx is not None:
            s[best_idx] = f"\\mathbf{{{s[best_idx]}}}"
            s[second_best_idx] = f"\\mathit{{{s[second_best_idx]}}}"

        return s

    # Fill missing rows of df_l2 with nan values
    for row in df_l1.index:
        if row not in df_l2.index:
            df_l2.loc[row] = np.nan

    # Fill missing columns of df_l2 with nan values
    for column in df_l1.columns:
        if column not in df_l2.columns:
            df_l2[column] = np.nan

    df = (
        "$"
        + df_l1.apply(highlight_best, axis=highlight_axis)
        + "_{"
        + df_l2.applymap(float_formatter)
        + "}$"
    )

    # Replace all $nan_{nan}$ by empty strings
    df = df.replace(r"(?i)\$nan_\{nan\}\$", "", regex=True)
    # Replace all _{nan} by empty strings
    df = df.replace(r"(?i)(.+)_{nan}\$", r"\1$", regex=True)

    return df
