from typing import Union, Optional
import polars as pl
import pandas as pd
import numpy as np
from .geometry_compensation import (
    determine_mpr_shaft_speed,
    get_mpr_geometry,
    perform_alignment_err,
)
from ..utils import _get_dataframe_library_preference
import duckdb


def calculate_ias(
    arr_toas: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
    N: int,
    M: int = 10,
    beta: Optional[float] = 1.0e10,
    sigma: Optional[float] = 10,
    M_recalibrate: float = 7.76,
    alignment_error_threshold_multiplier: float = 0.25,
) -> Union[pl.DataFrame, pd.DataFrame]:
    """
    Calculate instantaneous angular speed (IAS) using encoder geometry compensation.

    This function processes time-of-arrival (TOA) data from an incremental shaft encoder,
    calibrates the encoder geometry using Bayesian regression, and calculates instantaneous
    angular speeds. It implements a complete workflow consisting of geometry calibration,
    periodic recalibration, geometry pattern detection, and section alignment.

    Parameters
    ----------
    arr_toas :Union[np.ndarray, pd.DataFrame, pl.DataFrame]
        Time-of-arrival measurements from the encoder. If a Pandas or Polars DataFrame is
        supplied, the first column will be converted to a numpy array.
    N : int
        Number of sections in the encoder. This must match the physical encoder configuration.
    M : int
        The number of revolutions spanned by arr_toas. Defaults to 10.
    beta : float, optional
        The beta value for the Bayesian Geometry Compensation, being the precision of the likelihood function.
        A larger value is attributed to the observed values being approximately noise free.
        Defaults to 1.0e10.
    sigma : float, optional
        The sigma value for the Bayesian Geometry Compensation, being the standard deviation of the prior probability.
        A larger value is attributed to little confidence in prior probabilities.
        Defaults to 10.
    M_recalibrate : float, optional
        The number of revolutions after which the encoder
        should be recalibrated. Must be less than M. Default is 7.76 revolutions.
    alignment_error_threshold_multiplier : float, optional
        The multiplication factor to be multiplied with the absolute of
        the median of the alignment errors. Sometimes there is a clear
        issue in the alignment and not all sections in a signal are
        aligned. Often this is identified by long breaks in the time signal.

        Recommendation: set to 0.5 or 0.8 to prevent the algorithm
        not aligning sections as the error is too high. 
        Higher values (0.5-0.8) are more lenient for signals with gaps or noise,
        while lower values are stricter. Default is 0.25.

    Returns
    -------
    Union[pl.DataFrame, pd.DataFrame]
        A DataFrame containing the shaft speeds of the encoder.
        DataFrame containing detailed shaft speed and section information with columns:
    - section_start_time: Start time of each encoder section
    - section_end_time: End time of each encoder section
    - section_distance: Calibrated angular distance of each section in radians
    - Omega: Instantaneous angular velocity (rad/s) for each section
    - n: Revolution counter
    - section_start: Start angle of each section in radians
    - section_end: End angle of each section in radians
    
    The output format (Polars or Pandas) is determined by the global preference
    set with _get_dataframe_library_preference().

    Raises
    ------
    ValueError
        If M_recalibrate >= M, which would prevent proper recalibration.
        
    Notes
    -----
    This function combines multiple steps:
    1. Encoder geometry calibration using Bayesian regression
    2. Periodic recalibration across the dataset
    3. Detection of the true encoder geometry pattern 
    4. Alignment of each section to the detected pattern
    5. Calculation of instantaneous angular speed
    
    Section alignment may fail for portions of the signal with significant 
    noise or missing data. Such regions are filtered out in the final result.

    See Also
    --------
    determine_mpr_shaft_speed : Used for calibration and speed calculation
    get_mpr_geometry : Used to determine encoder geometry pattern
    perform_alignment_err : Used to align sections with the encoder pattern

    References
    ----------
    [1] D. H. Diamond, P. S. Heyns, and A. J. Oberholster, “Online Shaft Encoder Geometry
        Compensation for Arbitrary Shaft Speed Profiles Using Bayesian Regression,” Mechanical Systems
        and Signal Processing, vol. 81, pp. 402-418, Dec. 2016, doi: 10.1016/j.ymssp.2016.02.060.

    Copyright (C) 2016 Dawie Diamond, dawie.diamond@yahoo.com

    www.up.ac.za/caim
    www.caimlabs.co.za
    """

    if M_recalibrate >= M:
        raise ValueError(
            "M_recalibrate must be less than M. Try using the default values first"
            ", they are usually good enough."
        )
    if isinstance(arr_toas, (pd.DataFrame, pl.DataFrame)):
        arr_toas = duckdb.query(
            f"""SELECT {list(
            duckdb.execute(
                "SELECT * FROM arr_toas LIMIT 0"
            ).fetchnumpy().keys()
        )[0]} as toa FROM arr_toas"""
        ).fetchnumpy()["toa"]
    print("Calculating the IAS. This may take a while...")
    df_mpr = determine_mpr_shaft_speed(
        arr_toas, N=N, M=M, beta=beta, sigma=sigma, M_recalibrate=M_recalibrate
    )
    df_geometry = get_mpr_geometry(df_mpr, N)
    arr_is_new_revo, arr_sections_start, arr_sections_end = perform_alignment_err(
        df_mpr["section_distance"].to_numpy(),
        df_geometry["section_distance"].to_numpy(),
        df_geometry["section_start"].to_numpy(),
        df_geometry["section_end"].to_numpy(),
        alignment_error_threshold_multiplier,
    )
    df_mpr_speed = df_mpr.with_columns(
        [
            pl.Series("n", arr_is_new_revo).cum_sum().alias("n"),
            pl.Series("section_start", arr_sections_start),
            pl.Series("section_end", arr_sections_end),
        ]
    ).filter(pl.col("section_start") >= 0)
    if _get_dataframe_library_preference() == "pd":
        return df_mpr_speed.to_pandas()
    return df_mpr_speed
