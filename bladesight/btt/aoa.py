# Copyright (c) ARAMI Physical Asset Management Pty Ltd T/A Bladesight Inc. (2023)

from typing import List, Tuple, Union, Literal

import numpy as np
from numba import njit

import pandas as pd
import polars as pl

def get_numpy_series_from_df(
        df : Union[pd.DataFrame, pl.DataFrame],
        column_name : Union[str, int] 
    ) -> np.ndarray:
    """This function extracts a single column from a DataFrame.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): The DataFrame from which
            to extract the column.
        column_name Union[str, int]: The name or number of the column to
            extract.

    Returns:
        np.ndarray: The extracted column.
    """
    if isinstance(df, pd.DataFrame):
        if isinstance(column_name, int):
            return df.iloc[:, column_name].to_numpy()
        else:
            return df[column_name].to_numpy()
    elif isinstance(df, pl.DataFrame):        
        return df[:, column_name].to_numpy()

def get_df_return_lib(
        df : Union[pd.DataFrame, pl.DataFrame]
    ) -> Literal['pl', 'pd']:
    """This function returns the library of the DataFrame.

    Args:
        df (Union[pd.DataFrame, pl.DataFrame]): The DataFrame to
            determine the library of.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: The library of the DataFrame.
    """
    if isinstance(df, pd.DataFrame):
        return 'pd'
    elif isinstance(df, pl.DataFrame):
        return 'pl'

@njit
def calculate_aoa(arr_opr_zero_crossing: np.ndarray, arr_probe_toas: np.ndarray):
    """
    This function calculates the angle of arrival of
    each ToA value relative to the revolution in
    which it occurs.

    Args:
        arr_opr_zero_crossing (np.array): An array of
            OPR zero-crossing times.
        arr_probe_toas (np.array): An array of
            ToA values.

    Returns:
        np.array: A matrix of AoA values. Each row in the
            matrix corresponds to a ToA value. The columns
            are:
            0: The revolution number
            1: The zero crossing time at the start of the revolution
            2: The zero crossing time at the end of the revolution
            3: The angular velocity of the revolution
            4: The ToA
            5: The AoA of the ToA value
    """
    num_toas = len(arr_probe_toas)
    AoA_matrix = np.zeros((num_toas, 6))

    AoA_matrix[:, 0] = -1

    current_zero_crossing_start = arr_opr_zero_crossing[0]
    current_zero_crossing_end = arr_opr_zero_crossing[1]
    Omega = 2 * np.pi / (current_zero_crossing_end - current_zero_crossing_start)
    current_n = 0

    for i, toa in enumerate(arr_probe_toas):
        while toa > current_zero_crossing_end:
            current_n += 1
            if current_n >= (len(arr_opr_zero_crossing) - 1):
                break
            current_zero_crossing_start = arr_opr_zero_crossing[current_n]
            current_zero_crossing_end = arr_opr_zero_crossing[current_n + 1]
            Omega = (
                2 * np.pi / (current_zero_crossing_end - current_zero_crossing_start)
            )
        if current_n >= (len(arr_opr_zero_crossing) - 1):
            break

        if toa > current_zero_crossing_start:
            AoA_matrix[i, 0] = current_n
            AoA_matrix[i, 1] = current_zero_crossing_start
            AoA_matrix[i, 2] = current_zero_crossing_end
            AoA_matrix[i, 3] = Omega
            AoA_matrix[i, 4] = toa
            AoA_matrix[i, 5] = Omega * (toa - current_zero_crossing_start)

    return AoA_matrix

def transform_ToAs_to_AoAs(
    df_opr_zero_crossings: Union[pd.DataFrame, pl.DataFrame],
    df_probe_toas: Union[pd.DataFrame, pl.DataFrame],
) -> Union[pd.DataFrame, pl.DataFrame]:
    """This function transforms the ToA values to AoA values for a
    single probe, given the OPR zero-crossing times and the proximity
    probe's ToA values. It receives Pandas DataFrames, and also
    cleans up the ToAs that could not be converted.

    The timestamps are assumed to reside in the first column of
    each DataFrame.

    Args:
        df_opr_zero_crossings (Union[pd.DataFrame, pl.DataFrame]): A DataFrame with the
            OPR zero-crossing times.
        df_probe_toas (Union[pd.DataFrame, pl.DataFrame]): A DataFrame with the probe's
            ToA values.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: A DataFrame with the AoA values.
    """
    return_type = get_df_return_lib(df_opr_zero_crossings)
    AoA_matrix = calculate_aoa(
        get_numpy_series_from_df(
            df_opr_zero_crossings, 
            0
        ), get_numpy_series_from_df(
            df_probe_toas,
            0
        )
    )
    if return_type == 'pd':
        df_AoA = pd.DataFrame(
            AoA_matrix, columns=["n", "n_start_time", "n_end_time", "Omega", "ToA", "AoA"]
        )
        df_AoA = df_AoA[df_AoA["n"] != -1]
        df_AoA.reset_index(inplace=True, drop=True)
        return df_AoA
    elif return_type == 'pl':
        df_AoA = pl.DataFrame(
            {
                "n": AoA_matrix[:, 0],
                "n_start_time": AoA_matrix[:, 1],
                "n_end_time": AoA_matrix[:, 2],
                "Omega": AoA_matrix[:, 3],
                "ToA": AoA_matrix[:, 4],
                "AoA": AoA_matrix[:, 5],
            }
        )
        df_AoA = df_AoA.filter(df_AoA["n"] != -1)
        return df_AoA


##########################################################################
#                             MPR ENCODER                                #
##########################################################################
def calculate_aoa_from_mpr(
    arr_mpr_zero_crossing: np.ndarray,
    arr_probe_toas: np.ndarray,
    mpr_sections: int = 1,
) -> np.ndarray:
    """This function calculates the angle of arrival of
    each ToA value relative to the section and revolution in
    which it occurs when using an MPR encoder.

    Args:
        arr_mpr_zero_crossing (np.ndarray): An array of MPR
            zero-crossing times.
        arr_probe_toas (np.ndarray): An array of ToA values.
        mpr_sections (int, optional): The number of sections
            in the MPR encoder. Defaults to 1, in this case,
            this function will be treated as an OPR encoder.

    Returns:
        np.ndarray: A matrix of AoA values. Each row in the
            matrix corresponds to a ToA value. The columns
            are:
            0: The revolution number
            1: The section number
            2: The zero crossing time at the start of the revolution
            3: The zero crossing time at the end of the revolution
            4: The angular velocity of the revolution
            5: The ToA
            6: The AoA of the ToA value
    """
    num_toas = len(arr_probe_toas)
    AoA_matrix = np.zeros((num_toas, 7))
    rad_per_section = 2 * np.pi / mpr_sections
    AoA_matrix[:, 0] = -1

    current_zero_crossing_start = arr_mpr_zero_crossing[0]
    current_zero_crossing_end = arr_mpr_zero_crossing[1]
    Omega = rad_per_section / (current_zero_crossing_end - current_zero_crossing_start)
    current_n = 0
    current_revo = 0
    current_section = 0

    for i, toa in enumerate(arr_probe_toas):
        while toa > current_zero_crossing_end:
            current_n += 1
            if current_n >= (len(arr_mpr_zero_crossing) - 1):
                break
            current_zero_crossing_start = arr_mpr_zero_crossing[current_n]
            current_zero_crossing_end = arr_mpr_zero_crossing[current_n + 1]
            Omega = rad_per_section / (
                current_zero_crossing_end - current_zero_crossing_start
            )
            current_section += 1
            if current_section == mpr_sections:
                current_section = 0
                current_revo += 1

        if current_n >= (len(arr_mpr_zero_crossing) - 1):
            break

        if toa > current_zero_crossing_start:
            AoA_matrix[i, 0] = current_revo
            AoA_matrix[i, 1] = current_section
            AoA_matrix[i, 2] = current_zero_crossing_start
            AoA_matrix[i, 3] = current_zero_crossing_end
            AoA_matrix[i, 4] = Omega
            AoA_matrix[i, 5] = toa
            AoA_matrix[i, 6] = (
                Omega * (toa - current_zero_crossing_start)
                + current_section * rad_per_section
            )

    return AoA_matrix

def transform_ToAs_to_AoAs_mpr(
    df_mpr_zero_crossings: pd.DataFrame,
    df_probe_toas: pd.DataFrame
) -> pd.DataFrame:
    """This function transforms the ToA values to AoA values for a
    single probe, given the MPR-calculated DataFrame.

    The timestamps are assumed to reside in the first column of
    df_probe.

    Args:
        df_opr_zero_crossings (pd.DataFrame): The result of the 
            bladesight.ias.calculate_mpr function.
        df_probe_toas (pd.DataFrame): A DataFrame with the probe's
            ToAs.
    Returns:
        pd.DataFrame: A DataFrame containing the AoA values.
    """
    if df_mpr_zero_crossings.columns.to_list() != [
        'section_start_time',
        'section_end_time',
        'section_distance',
        'Omega',
        'n',
        'section_start',
        'section_end'
    ]:
        raise AssertionError(
            """The DataFrame must have the columns: 'section_start_time', """
            """ 'section_end_time', 'section_distance', 'Omega', 'n', """ 
            """ 'section_start', 'section_end'"""
        )
    df_AoAs = pd.merge_asof(
        df_probe_toas,
        df_mpr_zero_crossings,
        left_on=df_probe_toas.columns[0],
        right_on="section_start_time",
        direction='backward'
    ).dropna().reset_index(drop=True)

    df_AoAs["AoA"] = (
        (
            (
                df_AoAs["toa"] - df_AoAs["section_start_time"]
            ) 
            * df_AoAs["Omega"] 
            + df_AoAs["section_start"]
        ) % (2*np.pi)
    )
    return df_AoAs[['n', 'section_start_time', 'section_end_time', 'Omega', 'toa', 'AoA']].rename(
        columns={'toa': 'ToA'}
    )


##########################################################################
#                    TRANSFORM PROX AoAs to Blade AoAs                   #
##########################################################################


def calculate_Q(
    arr_aoas: np.ndarray, d_theta: float, N: int
) -> Tuple[float, np.ndarray]:
    """This function calculates the binning quality factor, Q, 
    given the AoA values, the number og blades and the bin offset.

    Args:
        arr_aoas (np.ndarray): The proximity probe AoA values.
        d_theta (float): The bin offset.
        N (int): The number of blades on the rotor.

    Returns:
        Tuple[float, np.ndarray]: The binning quality factor, Q, and
        the bin edges.
    """
    bin_edges = np.linspace(0 + d_theta, 2 * np.pi + d_theta, N + 1)
    Q = 0
    for b in range(N):
        left_edge = bin_edges[b]
        right_edge = bin_edges[b + 1]
        bin_centre = (left_edge + right_edge) / 2
        bin_mask = (arr_aoas > left_edge) & (arr_aoas <= right_edge)
        Q += np.sum((arr_aoas[bin_mask] - bin_centre) ** 2)
    if np.sum(arr_aoas < bin_edges[0]) > 0:
        # In other words, some values are less than bin_edges[0],
        # meaning it actually occurs at the end of the previous
        # revolution
        left_edge_last = bin_edges[N - 1]
        right_edge_last = bin_edges[N]
        bin_centre_last = (left_edge_last + right_edge_last) / 2
        bin_mask = arr_aoas <= bin_edges[0]
        Q += np.sum(((2 * np.pi - arr_aoas[bin_mask]) - bin_centre_last) ** 2)
    if np.sum(arr_aoas > bin_edges[-1]) > 0:
        left_edge_first = bin_edges[0]
        right_edge_first = bin_edges[1]
        bin_centre_first = (left_edge_first + right_edge_first) / 2
        bin_mask = arr_aoas > bin_edges[-1]
        Q += np.sum(((arr_aoas[bin_mask] - 2 * np.pi) - bin_centre_first) ** 2)
    return Q, bin_edges

def transform_prox_AoAs_to_blade_AoAs(
    df_prox: Union[pd.DataFrame, pl.DataFrame],
    B: int,
    d_theta_increments: int = 200,
    spurious_pulse_policy : Literal['drop', 'keep'] = 'drop'
) -> List[Union[pd.DataFrame, pl.DataFrame]]:
    """This function takes a DataFrame containing the AoA values of a proximity probe,
    and returns a list of DataFrame, each containing the AoA values of a single blade.

    Args:
        df_prox (Union[pd.DataFrame, pl.DataFrame]): The dataframe containing the AoA values
            of the proximity probe.
        B (int): The number of blades.
        d_theta_increments (int, optional): The number of increments.
        spurious_pulse_policy (Literal['drop', 'keep'], optional): The policy to handle
            spurious pulses. Defaults to 'drop'.

    Returns:
        List[Union[pd.DataFrame, pl.DataFrame]]: A list of dataframes, each containing the AoA
        values of a single blade.
    """
    return_type = get_df_return_lib(df_prox)

    d_thetas = np.linspace(-0.5 * np.pi / B, 1.5 * np.pi / B, d_theta_increments)
    arr_aoas = get_numpy_series_from_df(df_prox, "AoA")
    Qs = []
    optimal_Q, optimal_bin_edges, optimal_d_theta = np.inf, None, None
    for d_theta in d_thetas:
        Q, bin_edges = calculate_Q(arr_aoas, d_theta, B)
        if Q < optimal_Q:
            optimal_Q = Q * 1
            optimal_bin_edges = bin_edges
            optimal_d_theta = d_theta * 1
        Qs.append(Q)

    blade_dfs = []
    blade_median_AoAs = []
    for b in range(B):
        ix_bin = (df_prox["AoA"] > optimal_bin_edges[b]) & (
            df_prox["AoA"] <= optimal_bin_edges[b + 1]
        )
        if b == 0:
            ix_bin = ix_bin | (df_prox["AoA"] > optimal_bin_edges[-1])
            df_bin = (
                df_prox.loc[ix_bin].copy().reset_index(drop=True).sort_values("ToA")
            )

            ix_wrap = df_bin["AoA"] > optimal_bin_edges[-1]
            df_bin.loc[ix_wrap, "AoA"] = df_bin.loc[ix_wrap, "AoA"] - 2 * np.pi
        elif b == B - 1:
            ix_bin = ix_bin | (df_prox["AoA"] <= optimal_bin_edges[0])
            df_bin = (
                df_prox.loc[ix_bin].copy().reset_index(drop=True).sort_values("ToA")
            )

            ix_wrap = df_bin["AoA"] > optimal_bin_edges[-1]
            df_bin.loc[ix_wrap, "AoA"] = 2 * np.pi - df_bin.loc[ix_wrap, "AoA"]
        else:
            df_bin = (
                df_prox.loc[ix_bin].copy().reset_index(drop=True).sort_values("ToA")
            )
        blade_dfs.append(df_bin)
        blade_median_AoAs.append( df_bin["AoA"].median()  )
    blade_order = np.argsort(blade_median_AoAs)
    blade_dfs = [blade_dfs[i] for i in blade_order]
    if spurious_pulse_policy == 'keep':
        return blade_dfs
    # Drop revolutions that has more than one occurrence of a blade
    blade_dfs_to_return = []
    for i, df in enumerate(blade_dfs):
        n_count = df["n"].value_counts()
        spurious_revos = n_count[n_count > 1].index
        blade_dfs_to_return.append(df[~df["n"].isin(spurious_revos)])
    return blade_dfs_to_return
            