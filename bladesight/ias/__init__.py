import polars as pl
import numpy as np
from .geometry_compensation import (
    determine_mpr_shaft_speed, 
    get_mpr_geometry, 
    perform_alignment_err
)

def calculate_mpr(
        arr_toas: np.ndarray, 
        N: int, 
        M: int = 10, 
        beta : float = 1E10,
        sigma : float = 0.01,
        M_recalibrate : float = 7.76,
    ) -> pl.DataFrame:
    """ Calculate the shaft speed and corresponding sections
        of the MPR encoder.

    Args:
        arr_toas (np.ndarray): The time of arrivals of the encoder.
        N (int): The number of sections in the encoder.
        M (int): The number of revolutions spanned by arr_toas. Defaults to 10.
        beta (float, optional): The beta value for the 
            Bayesian Geometry Compensation. Defaults to 1E10.
        sigma (float, optional): The sigma value for the
            Bayesian Geometry Compensation. Defaults to 0.01.
        M_recalibrate (float, optional): The number of revolutions
            after which the encoder should be recalibrated. Defaults to 7.76.

    Returns:
        pl.DataFrame: A DataFrame containing the shaft 
            speeds of the encoder.
    """
    df_mpr = determine_mpr_shaft_speed(
        arr_toas,
        N = N,
        M = M,
        beta = beta,
        sigma = sigma,
        M_recalibrate = M_recalibrate
    )
    df_geometry = get_mpr_geometry(df_mpr, N)
    arr_is_new_revo, arr_sections_start, arr_sections_end = perform_alignment_err(
        df_mpr["section_distance"].to_numpy(),
        df_geometry["section_distance"].to_numpy(),
        df_geometry["section_start"].to_numpy(),
        df_geometry["section_end"].to_numpy()
    )
    df_mpr_speed = df_mpr.with_columns(
        [
            pl.Series("n",arr_is_new_revo).cum_sum().alias("n"),
            pl.Series("section_start", arr_sections_start),
            pl.Series("section_end", arr_sections_end),
        ]
    ).filter(
        pl.col("section_start") >= 0
    )
    return df_mpr_speed
