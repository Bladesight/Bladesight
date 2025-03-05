from typing import List
import pandas as pd
import numpy as np


def pivot_blade_AoAs_along_revolutions(
    prox_AoA_dfs: List[pd.DataFrame],
) -> pd.DataFrame:
    """
    Align angle-of-arrival (AoA) DataFrames along shaft revolutions.

    The input DataFrames are expected to be the output of the
    transform_prox_AoAs_to_blade_AoAs function, each containing
    time-of-arrival (ToA) and AoA columns for a single blade.

    Parameters
    ----------
    prox_AoA_dfs : List[pd.DataFrame]
        A list of DataFrames
        where each DataFrame contains the ToA and AoA values
        for a single blade from a proximity probes, containing columns
        "n", "ToA", and "AoA".

    Returns
    -------
    pd.DataFrame
        A DataFrame where every row contains the
        data pertaining to a single shaft revolution and every
        blade's ToA and AoA values are in its own column respectively.
        The DataFrame is aligned by revolution ("n"), with each blade's
        "ToA" and "AoA" as separate columns.
    """
    df_blades_aligned = prox_AoA_dfs[0]
    df_blades_aligned = df_blades_aligned.rename(
        columns={"ToA": "ToA_1", "AoA": "AoA_1"}
    )
    for i, df_blade in enumerate(prox_AoA_dfs[1:]):
        df_blades_aligned = df_blades_aligned.merge(
            df_blade[["n", "ToA", "AoA"]].rename(
                columns={"ToA": "ToA_" + str(i + 2), "AoA": "AoA_" + str(i + 2)}
            ),
            how="outer",
            on="n",
        )
    return df_blades_aligned


def create_stack_plot_df(df_blades_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Create a DataFrame that shows consecutive differences between blades for each revolution.

    For each adjacent pair of blade AoAs, compute the difference
    (blade i+1 AoA - blade i AoA). Additionally, handle the wrap-around
    from the last blade to the first blade by offsetting one sample and adding 2π.

    Parameters
    ----------
    df_blades_aligned : pd.DataFrame
        DataFrame containing angle-of-arrival columns ("AoA_i") for each
        blade and a revolution column "n". This is the output
        of pivot_blade_AoAs_along_revolutions.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the revolution index "n" and difference
        columns corresponding to each adjacent blade pair.
    """
    all_aoa_columns = sorted(
        [i for i in df_blades_aligned.columns if i.startswith("AoA_")]
    )
    B = len(all_aoa_columns)
    stack_plot_diffs = {}
    stack_plot_diffs["n"] = df_blades_aligned["n"].to_numpy()
    for blade_no in range(B - 1):
        farther_blade_name = all_aoa_columns[blade_no + 1]
        closer_blade_name = all_aoa_columns[blade_no]
        arr_blade_diffs = (
            df_blades_aligned[farther_blade_name] - df_blades_aligned[closer_blade_name]
        ).to_numpy()

        stack_plot_diffs[closer_blade_name] = arr_blade_diffs
    farther_blade_name = all_aoa_columns[0]
    closer_blade_name = all_aoa_columns[B - 1]
    arr_blade_diffs = (
        df_blades_aligned[farther_blade_name].to_numpy()[1:]
        + 2 * np.pi
        - df_blades_aligned[closer_blade_name].to_numpy()[:-1]
    )
    arr_blade_diffs = np.append(arr_blade_diffs, [None])
    stack_plot_diffs[closer_blade_name] = arr_blade_diffs
    return pd.DataFrame(stack_plot_diffs)


def shift_AoA_column_headings(
    aoa_column_headings: List[str], shift_by: int
) -> List[str]:
    """
    Shift the columns headings of the AoA
    such that the first column heading represents the first blade
    arriving at the first probe.

    Parameters
    ----------
    aoa_column_headings : List[str]
        List of values to be shifted
    shift_by : int
        Number of positions to shift the headings.

    Returns
    -------
    List[str]
        Shifted column headings.

    Raises
    ------
    ValueError
        If shift_by is >= the total number of columns.
    """
    if shift_by >= len(aoa_column_headings):
        raise ValueError(
            "shift_by must be less than the number blades in aoa_column_headings"
        )
    return list(aoa_column_headings)[shift_by:] + list(aoa_column_headings)[:shift_by]


def rename_df_columns_for_alignment(
    df_to_align: pd.DataFrame, global_column_headings: List[str], shift_by: int
) -> pd.DataFrame:
    """
    Rename and reorder columns to align them with a set of global headings.

    This function performs two tasks:
    1) Determine the mapping between the global column headings and the column headings `df_to_align`.
    2) Reorders and renames the DataFrame columns to match the global headings.

    Parameters
    ----------
    df_to_align : pd.DataFrame
        DataFrame whose columns should be renamed and reordered.
    global_column_headings : List[str]
        The column headings to which the columns in df_to_align should be mapped. This will normally be AoA or ToA column headings.
    shift_by : int
        The number of positions to shift the values in the array by.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed and reordered columns.
    """
    # Create a dictionary that maps the column headings in df_to_align
    # to the global column headings
    shifted_dataframe_columns = shift_AoA_column_headings(
        global_column_headings, shift_by
    )
    column_headings_to_rename = {
        local_col: global_col
        for local_col, global_col in zip(
            shifted_dataframe_columns, global_column_headings
        )
    }
    original_column_order = list(df_to_align.columns)
    df_to_align = df_to_align.rename(columns=column_headings_to_rename)
    return df_to_align[original_column_order]


def predict_probe_offset(
    df_probe_AoAs: pd.DataFrame,
    starting_aoa: float,
    prox_probe_relative_distance: float,
) -> int:
    """
    Compute the offset that needs to be applied to the AoA columns of the current probe to align them with the first
    probe.

    Parameters
    ----------
    df_probe_AoAs : pd.DataFrame
        A DataFrame where every row contains the data pertaining to a single shaft
        revolution and every blade's ToA and AoA values are in its
        own column respectively. This is the output of the
        `pivot_blade_AoAs_along_revolutions` function.
    starting_aoa : float
        The mean AoA of the blade you want to project forward and identify in df_probe_AoAs. In radians.
    prox_probe_relative_distance : float
        The relative distance between the current probe and the first probe. In radians.

    Returns
    -------
    int
        The blade offset that needs to be applied to the AoA values in df_probe_AoAs to align it to the blade in starting_aoa
    """
    predicted_blade_position = (starting_aoa + prox_probe_relative_distance) % (
        2 * np.pi
    )
    all_aoa_columns = sorted([i for i in df_probe_AoAs.columns if i.startswith("AoA_")])
    current_probe_median_AoAs = df_probe_AoAs[all_aoa_columns].median()
    err_aoa = np.abs(current_probe_median_AoAs - predicted_blade_position)
    offset = np.argmin(err_aoa)
    return offset


def assemble_rotor_AoA_dfs(
    prox_aligned_dfs: List[pd.DataFrame], probe_spacing: List[float]
) -> List[pd.DataFrame]:
    """
    Assemble the rotor blade AoA DataFrames. In other
    words, this function receives the grouped AoA DataFrames from each
    probe, the one calculated by `pivot_blade_AoAs_along_revolutions` and
    shifts the AoA values of each probe such that the first
    blade arriving at the first probe is aligned with the first blade
    arriving at the first probe.

    Parameters
    ----------
    prox_aligned_dfs : List[pd.DataFrame]
        A list of DataFrames where each DataFrame contains the ToAs and AoAs of a single
        blade from a proximity probe. Each DataFrame is the output of the `pivot_blade_AoAs_along_revolutions` function.
        The DataFrames should contain "ToA_i" and "AoA_i" columns.
    probe_spacing : List[float]
        A list of relative probe spacing
        between the first probe and every other probe. There are one
        less value in this list than in prox_aligned_dfs.

    Returns
    -------
    List[pd.DataFrame]
        A list of DataFrames where each DataFrame
        contains the ToAs and AoAs of a single blade over all
        the proximity probes.
    """
    all_aoa_columns = sorted(
        [i for i in prox_aligned_dfs[0].columns if i.startswith("AoA_")]
    )
    all_toa_columns = sorted(
        [i for i in prox_aligned_dfs[0].columns if i.startswith("ToA_")]
    )
    remaining_columns = [
        i
        for i in prox_aligned_dfs[0].columns
        if not i.startswith("ToA_") and not i.startswith("AoA_")
    ]
    B = len(all_aoa_columns)
    P = len(prox_aligned_dfs)
    if P - 1 != len(probe_spacing):
        raise ValueError(
            "The number of proximity probes must be "
            "one less than the number of probe spacings"
        )
    rotor_blade_dfs = []
    for b in range(1, B + 1):
        columns_to_copy = remaining_columns + [f"ToA_{b}", f"AoA_{b}"]
        rename_dict = {f"ToA_{b}": "ToA_p1", f"AoA_{b}": "AoA_p1"}
        rotor_blade_dfs.append(
            prox_aligned_dfs[0][columns_to_copy]
            .copy(deep=True)
            .rename(columns=rename_dict)
        )
    blade_1_probe_1_median = rotor_blade_dfs[0]["AoA_p1"].median()
    for iter_count, (df_probe_AoA, probe_offset) in enumerate(
        zip(prox_aligned_dfs[1:], probe_spacing)
    ):
        probe_no = iter_count + 2
        probe_offset = predict_probe_offset(
            df_probe_AoA, blade_1_probe_1_median, probe_offset
        )
        df_probe_AoAs_aligned = rename_df_columns_for_alignment(
            df_probe_AoA, all_aoa_columns, probe_offset
        )
        df_probe_AoAs_aligned = rename_df_columns_for_alignment(
            df_probe_AoAs_aligned, all_toa_columns, probe_offset
        )
        for b in range(1, B + 1):
            columns_to_merge = ["n", f"ToA_{b}", f"AoA_{b}"]
            rename_dict = {
                f"ToA_{b}": f"ToA_p{probe_no}",
                f"AoA_{b}": f"AoA_p{probe_no}",
            }
            rotor_blade_dfs[b - 1] = rotor_blade_dfs[b - 1].merge(
                df_probe_AoAs_aligned[columns_to_merge].rename(columns=rename_dict),
                how="inner",
                on="n",
            )
    return rotor_blade_dfs