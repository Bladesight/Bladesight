from typing import Union
import polars as pl
import pandas as pd
import numpy as np
from .geometry_compensation import (
    determine_mpr_shaft_speed, 
    get_mpr_geometry, 
    perform_alignment_err
)
from ..utils import _get_dataframe_library_preference
import duckdb

def calculate_ias(
        arr_toas: Union[np.ndarray, pd.DataFrame, pl.DataFrame],
        N: int, 
        M: int = 10, 
        beta : float = 1E10,
        sigma : float = 0.01,
        M_recalibrate : float = 7.76,
        alignment_error_threshold_multiplier: float = 0.25
    ) -> Union[pl.DataFrame, pd.DataFrame]:
    """ Calculate the shaft speed and corresponding sections
        of the MPR encoder.

    Parameters
    ----------
        arr_toas :Union[np.ndarray, pd.DataFrame, pl.DataFrame]
            The time of arrivals of the encoder. If a Pandas or Polars DataFrame is
            supplied, the first column will be converted to a numpy array.
        N : int
            The number of sections in the encoder.
        M : int
            The number of revolutions spanned by arr_toas. Defaults to 10.
        beta : float, optional
            The beta value for the Bayesian Geometry Compensation. Defaults 
            to 1E10.
        sigma : float, optional
            The sigma value for the Bayesian Geometry 
            Compensation. Defaults to 0.01.
        M_recalibrate : float, optional
            The number of revolutions after which the encoder 
            should be recalibrated. Defaults to 7.76.
        alignment_error_threshold_multiplier : float, optional 
            The multiplication factor to be multiplied with the absolute of 
            the median of the alignment errors. Sometimes there is a clear 
            issue in the alignment and not all sections in a signal are 
            aligned. Often this is identified by long breaks in the time signal. 
            
            Recommendation: set to 0.5 or 0.8 to prevent the algorithm 
            not aligning sections as the error is too high. Defaults to 0.25.

    Returns
    -------
        Union[pl.DataFrame, pd.DataFrame]
            A DataFrame containing the shaft speeds of the encoder.
    """
    if M_recalibrate >= M:
        raise ValueError(
            "M_recalibrate must be less than M. Try using the default values first"
            ", they are usually good enough."
        )
    if isinstance(arr_toas, (pd.DataFrame, pl.DataFrame)):
        arr_toas = duckdb.query(f"""SELECT {list(
            duckdb.execute(
                "SELECT * FROM arr_toas LIMIT 0"
            ).fetchnumpy().keys()
        )[0]} as toa FROM arr_toas""").fetchnumpy()["toa"]
    print("Calculating the IAS. This may take a while...")
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
        df_geometry["section_end"].to_numpy(),
        alignment_error_threshold_multiplier
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
    if _get_dataframe_library_preference() == "pd":
        return df_mpr_speed.to_pandas()
    return df_mpr_speed
