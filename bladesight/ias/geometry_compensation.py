import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import polars as pl
from tqdm import tqdm
from numba import njit
from typing import Union, List, Optional


def perform_bayesian_geometry_compensation(
    t: np.ndarray,
    N: int,
    M: int,
    e: Union[np.ndarray, List] = [],
    beta: Optional[float] = 1.0e10,
    sigma: Optional[float] = 10,
) -> np.ndarray:
    """
    Perform geometry compensation on an incremental shaft encoder,
    with N sections measured over M revolutions, using Bayesian regression.

    This algorithm compensates for geometric irregularities in incremental shaft encoders
    under arbitrary shaft speed profiles by applying a Bayesian regression technique.

    Please reference the paper discussing the algorithm as
    described in the "References" section.

    Parameters
    ----------
    t : np.ndarray
        1D numpy array of zeros crossing times. The first zero crossing
        time indicates the start of the first section.  This array should therefore
        have exactly M*N + 1 elements.
    N : int
        The number of sections in the shaft encoder.
    M : int
        The number of complete revolutions over which
        the compensation must be performed.
    e : np.ndarray
        Initial estimate for the encoder geometry. If empty, all sections are 
        assumed equal (2π/N). Default is empty list.
    beta : float, optional
        Precision of the likelihood function.
        A larger value is attributed to the observed values being approximately noise free.
        Defaults to 1.0e10.
    sigma : float, optional
        Standard deviation of the prior probability.
        A larger value is attributed to little confidence in prior probabilities.
        Defaults to 10.

    Returns
    -------
    np.ndarray
        Array containing the circumferential distances of all N sections, normalized
        to sum to 2π radians.

    References
    ----------
    [1] D. H. Diamond, P. S. Heyns, and A. J. Oberholster, “Online Shaft Encoder Geometry
        Compensation for Arbitrary Shaft Speed Profiles Using Bayesian Regression,” Mechanical Systems
        and Signal Processing, vol. 81, pp. 402-418, Dec. 2016, doi: 10.1016/j.ymssp.2016.02.060.

    Copyright (C) 2016 Dawie Diamond, dawie.diamond@yahoo.com

    www.up.ac.za/caim
    www.caimlabs.co.za

    """
    if len(t) != M * N + 1:
        raise ValueError(
            "Input Error: The vector containing the zero-crossing times should contain exactly N*M + 1 values"
        )
    if len(e) != 0 and len(e) != N:
        raise ValueError(
            "Input Error The encoder input should either be an empty list or a list with N elements"
        )
    A = np.zeros((2 * M * N - 1, N + 2 * M * N))
    B = np.zeros((2 * M * N - 1, 1))

    T = np.ediff1d(t)

    A[0, :N] = np.ones(N)
    B[0, 0] = 2 * np.pi
    deduct = 0
    for m in range(M):
        if m == M - 1:
            deduct = 1
        for n in range(N - deduct):
            nm = m * N + n
            A[1 + nm, n] = 3.0
            A[1 + nm, N + nm * 2] = -1.0 / 2 * T[nm] ** 2
            A[1 + nm, N + nm * 2 + 1] = -2 * T[nm]
            A[1 + nm, N + (nm + 1) * 2 + 1] = -1 * T[nm]
    deduct = 0
    for m in range(M):
        if m == M - 1:
            deduct = 1
        for n in range(N - deduct):
            nm = m * N + n
            A[M * N + nm, n] = 6.0
            A[M * N + nm, N + nm * 2] = -2 * T[nm] ** 2
            A[M * N + nm, N + (nm + 1) * 2] = -1 * T[nm] ** 2
            A[M * N + nm, N + nm * 2 + 1] = -6 * T[nm]

    # Initialize prior vector
    m0 = np.zeros((N + 2 * M * N, 1))

    Sigma0 = np.identity(N + 2 * M * N) * sigma**2

    if len(e) == 0:
        eprior = np.ones(N) * 2 * np.pi / N
    else:
        eprior = np.array(e) * 1.0
    m0[:N, 0] = eprior * 1.0
    for m in range(M):
        for n in range(N):
            nm = m * N + n
            m0[N + nm * 2 + 1, 0] = m0[n, 0] / T[nm]

    SigmaN = Sigma0 + beta * A.T.dot(A)
    BBayes = Sigma0.dot(m0) + beta * A.T.dot(B)
    mN = np.array([spsolve(csr_matrix(SigmaN), csr_matrix(BBayes))]).T
    # Normalize encoder increments to add up to 2 pi
    epost = mN[:N, 0] * 2 * np.pi / (np.sum(mN[:N, 0]))
    return epost


def determine_mpr_speed_for_zero_crossings(
    arr_toas: np.ndarray,
    N: int,
    M: int,
    beta: Optional[float] = 1.0e10,
    sigma: Optional[float] = 10,
) -> pl.DataFrame:
    """
    Calculate shaft speeds using zero crossing times after encoder geometry calibration.
    
    This function applies Bayesian geometry compensation to calibrate the encoder geometry
    and then calculates instantaneous shaft speeds at each encoder section.

    Parameters
    ----------
    arr_toas : np.ndarray
        Array of time-of-arrivals (zero crossings) from the encoder.
    N : int
        The number of sections in the encoder.
    M : int
        The number of revolutions spanned by arr_toas.
    beta : float, optional
        The beta value for the Bayesian Geometry Compensation, being the precision of the likelihood function.
        A larger value is attributed to the observed values being approximately noise free.
        Defaults to 1.0e10.
    sigma : float, optional
        The sigma value for the Bayesian Geometry Compensation, being the standard deviation of the prior probability.
        A larger value is attributed to little confidence in prior probabilities.
        Defaults to 10.

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - section_start_time: Start time of each encoder section
        - section_end_time: End time of each encoder section  
        - encoder_section_rad: Calibrated angular distance of each section in radians
    Raises
    ------
    ValueError
        If the length of arr_toas is not equal to N*M + 1.
        
    See Also
    --------
    perform_bayesian_geometry_compensation : The underlying function used for geometry calibration.
    """
    if arr_toas.shape[0] != N * M + 1:
        raise ValueError("The length of arr_toas must be equal to N*M + 1")
    epost_BGC = perform_bayesian_geometry_compensation(
        arr_toas, N=N, M=M, beta=beta, sigma=sigma
    )
    arr_start_times = arr_toas[:-1].copy()
    arr_end_times = arr_toas[1:].copy()
    arr_section_distances = np.arange(N * M, dtype=np.float64)
    for m in range(M):
        arr_section_distances[m * N : (m + 1) * N] = epost_BGC * 1
    return pl.DataFrame(
        {
            "section_start_time": arr_start_times,
            "section_end_time": arr_end_times,
            "encoder_section_rad": arr_section_distances,
        }
    )


def determine_mpr_shaft_speed(
    arr_toas: np.ndarray,
    N: int,
    M: int,
    beta: Optional[float] = 1.0e10,
    sigma: Optional[float] = 10,
    M_recalibrate: Optional[int] = 7.76,
) -> pl.DataFrame:
    """
    Calculate shaft speeds with periodic encoder geometry recalibration.
    
    This function calibrates the encoder geometry across multiple segments of the signal
    and calculates shaft speeds. It uses a sliding window approach with periodic 
    recalibration to account for any drift or changes in the system over time.

    Parameters
    ----------
    arr_toas : np.ndarray
        Array of time-of-arrivals (zero crossings) from the encoder.
    N : int
        The number of sections in the encoder.
    M : int
        The number of revolutions spanned by arr_toas.
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
        should be recalibrated. Defaults to 7.76 revolutions.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing the shaft speeds of the encoder.
        DataFrame with columns:
        - section_start_time: Start time of each encoder section
        - section_end_time: End time of each encoder section
        - section_distance: Median calibrated angular distance of each section in radians
        - Omega: Instantaneous angular velocity (rad/s) calculated for each section
        
    Notes
    -----
    This function uses multiple calibration windows and takes the median of measurements
    for each section to improve robustness against noise and temporary variations.
    
    See Also
    --------
    determine_mpr_speed_for_zero_crossings : Used for individual calibration windows.
    perform_bayesian_geometry_compensation : The underlying calibration algorithm.
    """
    len_t = len(arr_toas)
    recalibrate_interval = int(M_recalibrate * N)

    section_measurements = {}
    for i_start in tqdm(np.array(range(0, len_t, recalibrate_interval))):
        i_end = i_start + N * M + 1
        if i_end > len_t:
            break
        df_cal = determine_mpr_speed_for_zero_crossings(
            arr_toas[i_start:i_end], N=N, M=M, beta=beta, sigma=sigma
        )
        for row_cal in df_cal.iter_rows(named=True):
            key = (row_cal["section_start_time"], row_cal["section_end_time"])
            if key in section_measurements:
                section_measurements[key].append(row_cal["encoder_section_rad"])
            else:
                section_measurements[key] = [row_cal["encoder_section_rad"]]

    df_section_measurements = pl.DataFrame(
        {
            "section_start_time": [key[0] for key in section_measurements.keys()],
            "section_end_time": [key[1] for key in section_measurements.keys()],
            "encoder_section_rads": [val for val in section_measurements.values()],
        },
        schema={
            "section_start_time": pl.Float64,
            "section_end_time": pl.Float64,
            "encoder_section_rads": pl.List(pl.Float64),
        },
    )
    return (
        df_section_measurements.with_columns(
            pl.col("encoder_section_rads").list.median().alias("section_distance")
        )
        .with_columns(
            [
                (
                    pl.col("section_distance")
                    / (pl.col("section_end_time") - pl.col("section_start_time"))
                ).alias("Omega")
            ]
        )
        .select(pl.exclude("encoder_section_rads"))
    )


def get_mpr_geometry(df_mpr_speed: pl.DataFrame, N: int) -> pl.DataFrame:
    """
    Calculate the geometry of the MPR encoder by analyzing section patterns.
    
    This function identifies unique encoder sections by analyzing multiple samples
    of N consecutive sections. It aligns these samples based on notable features 
    (like maximum deviation from median) to reconstruct the physical geometry of 
    the encoder.
    
    Parameters
    ----------
    df_mpr_speed : pl.DataFrame
        The DataFrame containing the blade AoAs. This is the
        result from the determine_mpr_shaft_speed function.
    N : int
        The number of sections in the encoder.

    Returns
    -------
    pl.DataFrame
        The DataFrame containing the geometry
        of the MPR encoder.
        DataFrame with columns:
        - section_start: Start angle of each section in radians
        - section_end: End angle of each section in radians
        - section_distance: Angular distance of each section in radians

    Notes
    -----
    The function samples multiple sequences of N consecutive sections and aligns them
    to identify the true encoder geometry. This handles variations by taking the median
    of multiple measurements for each section.
    
    The returned geometry is normalized so that the full encoder spans exactly 2π radians.
    """
    n_samples_total = df_mpr_speed.height
    n_random_samples = min(int(n_samples_total / 10), 50)
    random_alignment_start_points = df_mpr_speed.select(
        pl.arange(10 * N, pl.count("Omega") - 10 * N)
    ).sample(n_random_samples)

    df_encoder_order = None
    for n, start in enumerate(random_alignment_start_points[:, 0]):
        df_one_section = (
            df_mpr_speed[start : start + N]
            .select("section_distance")
            .with_columns(
                [
                    pl.col("section_distance"),
                    (pl.col("section_distance") - pl.col("section_distance").median())
                    .abs()
                    .alias("max_abs_section"),
                ]
            )
            .with_columns(
                [
                    pl.when(
                        pl.col("max_abs_section") == pl.col("max_abs_section").max()
                    )
                    .then(pl.lit(1))
                    .otherwise(pl.lit(None))
                    .alias("is_max_asb_section")
                    .fill_null(strategy="forward")
                ]
            )
            .with_columns(
                (
                    (
                        pl.arange(
                            pl.col("is_max_asb_section").cum_sum().max(),
                            pl.col("is_max_asb_section").cum_sum().max() + N,
                        )
                    )
                    % N
                ).alias("section_no")
            )
        ).sort("section_no")

        if df_encoder_order is None:
            df_encoder_order = df_one_section.select(
                pl.col("section_distance").alias(f"section_distance_{n}")
            )
        else:
            df_encoder_order = df_encoder_order.hstack(
                df_one_section.select(
                    pl.col("section_distance").alias(f"section_distance_{n}")
                )
            )
    df_geometry = (
        pl.DataFrame(
            {"section_distance": np.median(df_encoder_order.to_numpy(), axis=1)}
        )
        .with_columns(
            pl.col("section_distance").cum_sum().alias("section_end"),
        )
        .with_columns(
            ((pl.col("section_end") / pl.col("section_end").max()) * (2 * np.pi)).alias(
                "section_end"
            )
        )
        .with_columns(
            (pl.col("section_end") - pl.col("section_distance")).alias("section_start")
        )
        .select("section_start", "section_end", "section_distance")
    )
    df_geometry[0, "section_start"] = 0
    return df_geometry


@njit
def perform_alignment_err(
    arr_sections: np.array,
    arr_geometry: np.array,
    arr_geometry_start: np.array,
    arr_geometry_end: np.array,
    alignment_error_threshold_multiplier: float = 0.25,
) -> np.ndarray:
    """
    Align encoder sections with a reference geometry and detect revolution starts.
    
    This function identifies where new revolutions begin in the encoder signal by
    comparing segments of the sections array against a reference geometry. It calculates
    alignment errors and uses a threshold to determine valid alignments.

    This will be used to determine the best alignment between the
    start of each revolution.

    Parameters
    ----------
    arr_sections : np.array
        The MPR sections array. This is the
        'section_distance' column from the determine_mpr_shaft_speed
        function.
    arr_geometry : np.array
        The geometry array. This is the
        'section_distance' column from the determine_geometry function.

    arr_geometry_start : np.array
        The start of the geometry array.

    arr_geometry_end : np.array
        The end of the geometry

    alignment_error_threshold_multiplier : float, optional
        The multiplication factor to be multiplied with the absolute of
        the median of the alignment errors. Sometimes there is a clear
        issue in the alignment and not all sections in a signal are
        aligned. Often this is identified by long breaks in the time signal.
        Recommendation: set to 0.5 or 0.8 to prevent the algorithm not
        aligning sections as the error is too high. Defaults to 0.25.

    Returns
    -------
    arr_is_new_revo : np.ndarray
            Binary array where 1 indicates the start of a new revolution.
    arr_sections_start : np.ndarray
        Array with the start angles for each section.
    arr_sections_end : np.ndarray
        Array with the end angles for each section.
    
    Notes
    -----
    This function uses a sliding window approach to compare sequences of sections
    with the reference geometry. When a close match is found (error below threshold),
    it marks the start of a new revolution and assigns the correct angular positions.
    
    For segments that don't align well (e.g., due to missing data or noise),
    the start/end values remain as -1.
    
    This implementation is JIT-compiled with Numba for performance.
    """
    arr_errors = np.ones_like(arr_sections) * -1
    N = len(arr_geometry)
    for i in np.arange(arr_sections.shape[0] - N):
        arr_errors[i] = np.sum(np.abs(arr_sections[i : i + N] - arr_geometry))
    arr_err_threshold = np.median(arr_errors) * alignment_error_threshold_multiplier
    arr_is_new_revo = np.zeros_like(arr_sections, dtype=np.int8)
    arr_sections_start = np.ones_like(arr_sections) * -1
    arr_sections_end = np.ones_like(arr_sections) * -1
    for i in np.arange(arr_sections.shape[0] - N):
        if arr_errors[i] < arr_err_threshold and arr_errors[i] != -1:
            arr_is_new_revo[i] = 1
            arr_sections_start[i : i + N] = arr_geometry_start
            arr_sections_end[i : i + N] = arr_geometry_end
    return arr_is_new_revo, arr_sections_start, arr_sections_end
