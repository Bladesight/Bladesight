# Copyright (c) ARAMI Physical Asset Management Pty Ltd T/A Bladesight Inc. (2023)

from typing import List, Tuple, Union, Literal

import numpy as np
from numba import njit
import duckdb

import pandas as pd
import polars as pl

from ..utils import _get_dataframe_library_preference


def transform_ToAs_to_AoAs(
    df_opr_zero_crossings: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    df_probe_toas: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Transform the Time of Arrival (ToA) values to Angle of Arrival (AoA) values for a
    single probe, given the OPR zero-crossing times and the proximity
    probe's ToA values. It receives Pandas DataFrames, and also
    cleans up the ToAs that could not be converted.

    Parameters
    ----------
    df_opr_zero_crossings : Union[np.ndarray, pd.DataFrame, pl.DataFrame]
        OPR zero-crossing timestamps. If a DataFrame is passed, the first column
        is assumed to contain the timestamps.
    df_probe_toas : Union[np.ndarray, pd.DataFrame, pl.DataFrame]
        Proximity probe ToAs. If a DataFrame is passed, the first column
        is assumed to contain the timestamps.

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame containing AoA values. Library depends on
        user’s DataFrame preference (Pandas or Polars).

    Raises
    ------
    ValueError
        If either df_opr_zero_crossings or df_probe_toas is invalid or if the
        preferred DataFrame library is not 'pd' or 'pl'.
    """
    if not isinstance(df_opr_zero_crossings, (np.ndarray, pd.DataFrame, pl.DataFrame)):
        raise ValueError("df_opr_zero_crossings must be a numpy array or a DataFrame.")
    if not isinstance(df_probe_toas, (np.ndarray, pd.DataFrame, pl.DataFrame)):
        raise ValueError("df_probe_toas must be a numpy array or a DataFrame.")
    if isinstance(df_opr_zero_crossings, np.ndarray):
        df_opr_zero_crossings = pl.DataFrame({"toas": df_opr_zero_crossings})
    if isinstance(df_probe_toas, np.ndarray):
        df_probe_toas = pl.DataFrame({"toas": df_probe_toas})
    first_column_opr = (
        duckdb.execute("SELECT * FROM df_opr_zero_crossings LIMIT 0").pl().columns[0]
    )

    first_column_probe = (
        duckdb.execute("SELECT * FROM df_probe_toas LIMIT 0").pl().columns[0]
    )

    query = f"""
    WITH TAB1 AS (
        SELECT 
            (row_number() OVER () - 1)::UINTEGER  as n,
            {first_column_opr} AS n_start_time,
            LEAD({first_column_opr}) OVER (ORDER BY {first_column_opr}) AS n_end_time,
            2 * PI()/(n_end_time - n_start_time) AS Omega
        FROM 
            df_opr_zero_crossings
    )

    SELECT 
        TAB1.*,
        df_probe_toas.{first_column_probe} AS ToA,
        (
            df_probe_toas.{first_column_probe} 
            - TAB1.n_start_time
        ) * Omega AS AoA,
    FROM df_probe_toas
    LEFT JOIN TAB1 
    ON df_probe_toas.{first_column_probe} >= TAB1.n_start_time
    WHERE 
        Omega IS NOT NULL
    AND
        ToA <= n_end_time
    """
    if _get_dataframe_library_preference() == "pd":
        return duckdb.execute(query).df()
    elif _get_dataframe_library_preference() == "pl":
        return duckdb.execute(query).pl()
    else:
        raise ValueError(
            "The DataFrame library preference must be either 'pd' or 'pl'."
            " Please set it using the bladesight.utils.set_dataframe_library_preference function."
        )


def transform_ToAs_to_AoAs_mpr(
    df_ias: Union[pd.DataFrame, pl.DataFrame],
    df_probe_toas: Union[pd.DataFrame, pl.DataFrame],
) -> Union[pd.DataFrame, pl.DataFrame]:
    """
    Transform Time of Arrival (ToA) data into Angle of Arrival (AoA) for a single probe using MPR data.

    Parameters
    ----------
    df_ias : Union[pd.DataFrame, pl.DataFrame]
        The result of bladesight.ias.calculate_ias containing MPR data.
    df_probe_toas : Union[pd.DataFrame, pl.DataFrame]
        DataFrame with the probe's ToAs in its first column.

    Returns
    -------
    Union[pd.DataFrame, pl.DataFrame]
        DataFrame containing ToA and AoA columns. The choice of Pandas or Polars
        depends on user preference.

    Raises
    ------
    TypeError
        If df_ias or df_probe_toas is not a DataFrame.
    AssertionError
        If df_ias does not match the required MPR schema.
    ValueError
        If the DataFrame library preference is invalid.
    """
    if not isinstance(df_ias, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("df_ias must be a Pandas or Polars DataFrame.")
    if not isinstance(df_probe_toas, (pd.DataFrame, pl.DataFrame)):
        raise TypeError("df_probe_toas must be a Pandas or Polars DataFrame.")
    present_columns = duckdb.execute("SELECT * FROM df_ias LIMIT 0").pl().columns

    if present_columns != [
        "section_start_time",
        "section_end_time",
        "section_distance",
        "Omega",
        "n",
        "section_start",
        "section_end",
    ]:
        raise AssertionError(
            """The DataFrame must have the columns: 'section_start_time', """
            """ 'section_end_time', 'section_distance', 'Omega', 'n', """
            """ 'section_start', 'section_end'"""
        )
    left_merge_key = (
        duckdb.execute("SELECT * FROM df_probe_toas LIMIT 0").pl().columns[0]
    )
    query = f"""
        SELECT 
            ias.n,
            ias.section_start_time,
            ias.section_end_time,
            ias.Omega,
            probe.toa as ToA,
            ((probe.toa - ias.section_start_time) * ias.Omega + ias.section_start) % (2*PI()) as AoA
        FROM df_probe_toas probe
        ASOF LEFT JOIN 
            df_ias ias
            ON {left_merge_key} >= section_start_time
        """
    if _get_dataframe_library_preference() == "pl":
        return duckdb.execute(query).pl()
    elif _get_dataframe_library_preference() == "pd":
        return duckdb.execute(query).df()
    else:
        raise ValueError(
            "The DataFrame library preference must be either 'pd' or 'pl'."
            " Please set it using the bladesight.utils.set_dataframe_library_preference function."
        )


##########################################################################
#                    TRANSFORM PROX AoAs to Blade AoAs                   #
##########################################################################


def calculate_Q(
    arr_aoas: Union[np.ndarray, pd.Series, pl.Series], d_theta: float, N: int
) -> Tuple[float, np.ndarray]:
    """
    Calculate the binning quality factor, Q, given Angle of Arrival (AoA) values, rotor blade count, and bin offset.

    Parameters
    ----------
    arr_aoas : Union[np.ndarray, pd.Series, pl.Series]
        Proximity probe AoA values. If a Pandas Series or Polars Series is passed, the values
            are extracted.
    d_theta : float
        Bin offset (in radians).
    N : int
        Number of blades on the rotor.

    Returns
    -------
    Tuple[float, np.ndarray]
        Q : float
            The binning quality factor. Q is the sum of squared errors between the center of each bin and each AoA within it.
        bin_edges : np.ndarray
            Bin edges used in the Q calculation.
    """
    if isinstance(arr_aoas, (pd.Series, pl.Series)):
        arr_aoas = arr_aoas.to_numpy()
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
    spurious_pulse_policy: Literal["drop", "keep"] = "drop",
) -> List[Union[pd.DataFrame, pl.DataFrame]]:
    """
    Convert proximity probe Angle of Arrivals (AoAs) into per-blade AoAs by optimal binning. 
    This function takes a DataFrame containing the AoA values of a proximity probe,
    and returns a list of DataFrame, each containing the AoA values of a single blade.

    Parameters
    ----------
    df_prox : Union[pd.DataFrame, pl.DataFrame]
        DataFrame containing the probe's AoA measurements.
    B : int
        Number of blades.
    d_theta_increments : int, optional
        Number of increments. The default is 200.
    spurious_pulse_policy : {'drop', 'keep'}, optional
        Policy to handle spurious pulses, by default 'drop'.

    Returns
    -------
    List[Union[pd.DataFrame, pl.DataFrame]]
        A list of DataFrames, each with AoA values corresponding to a single blade.

    Notes
    -----
    Any revolutions with more than one occurrence of a blade are dropped
    unless spurious_pulse_policy is set to 'keep'.
    """
    return_type = _get_dataframe_library_preference()

    d_thetas = np.linspace(-0.5 * np.pi / B, 1.5 * np.pi / B, d_theta_increments)
    arr_aoas = duckdb.execute("SELECT AoA FROM df_prox").fetchnumpy()["AoA"]
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
        if b == 0:
            query = f"""
                SELECT 
                    * 
                FROM df_prox
                WHERE
                    (
                        (AoA > {optimal_bin_edges[b]})
                            AND
                        (AoA <= {optimal_bin_edges[b + 1]})
                    )
                OR (
                    AoA > {optimal_bin_edges[-1]}
                )
                ORDER BY ToA
            """
        elif b == B - 1:
            query = f"""
                WITH TAB1 AS (
                    SELECT 
                        * 
                    FROM df_prox
                    WHERE
                        (
                            (AoA > {optimal_bin_edges[b]})
                                AND
                            (AoA <= {optimal_bin_edges[b + 1]})
                        )
                    OR (
                        AoA <= {optimal_bin_edges[0]}
                    )
                    ORDER BY ToA
                )
                SELECT
                    * EXCLUDE (AoA),
                    CASE
                        WHEN AoA > {optimal_bin_edges[-1]} THEN 2 * PI() - AoA
                        ELSE AoA
                    END AS AoA
                FROM TAB1
            """
        else:
            query = f"""
                SELECT 
                    * 
                FROM df_prox
                WHERE
                    (
                        (AoA > {optimal_bin_edges[b]})
                        AND
                        (AoA <= {optimal_bin_edges[b + 1]})
                    )
                ORDER BY ToA
            """
        if return_type == "pd":
            df_bin = duckdb.execute(query).df()
        else:
            df_bin = duckdb.execute(query).pl()
        blade_dfs.append(df_bin)
        blade_median_AoAs.append(
            duckdb.execute("SELECT AoA.MEDIAN() as median FROM df_bin").fetchone()[0]
        )

    blade_order = np.argsort(blade_median_AoAs)
    blade_dfs = [blade_dfs[i] for i in blade_order]

    if spurious_pulse_policy == "keep":
        return blade_dfs

    # Drop revolutions that has more than one occurrence of a blade
    blade_dfs_to_return = []
    query = """
        SELECT * FROM df
        QUALIFY row_number() OVER (PARTITION BY n ORDER BY ToA) = 1
        ORDER BY n, ToA
    """
    for _, df in enumerate(blade_dfs):
        if return_type == "pd":
            blade_dfs_to_return.append(duckdb.execute(query).df())
        else:
            blade_dfs_to_return.append(duckdb.execute(query).pl())
    return blade_dfs_to_return
