from typing import Optional

import numpy as np
from numba import njit

@njit
def threshold_crossing_interp(
    arr_t: np.ndarray,
    arr_s: np.ndarray,
    threshold: float,
    n_est: Optional[float] = None,
    trigger_on_rising_edge: bool = True,
) -> np.ndarray:
    """
    Sequentially detect threshold crossings in a signal and calculate Time of Arrival (ToA).

    If the threshold is exceeded (on rising edge) or gone below (on falling edge),
    the function interpolates between the two samples around the threshold
    to determine the precise crossing time.

    Parameters
    ----------
    arr_t : np.ndarray
        Time array corresponding to each sample in arr_s.
    arr_s : np.ndarray
        Signal voltage array corresponding to the time values.
    threshold : float
        Threshold value in the same units as arr_s.
    n_est : float, optional
        The estimated number of ToAs in this signal.
        Defaults to None.
        This number is used to pre-allocate the array containing the
        ToAs. If this number is not provided, the array will be pre-allocated as
        the same dimension as arr_t and arr_s.
    trigger_on_rising_edge : bool, optional
        If True, triggers on the rising edge (threshold crossing from below
        to above). If False, triggers on falling edge. Defaults to True.

    Returns
    -------
    np.ndarray
        Array of interpolated ToAs. Its length is determined by the
        number of threshold crossings or n_est.

    Raises
    ------
    ValueError
        If the number of detected crossings exceeds the pre-allocated size.
    """
    # Pre-allocate the array containing the ToAs
    if n_est is None:
        arr_toa = -1 * np.ones(arr_t.shape)
    else:
        arr_toa = -1 * np.ones(n_est)
    
    # Initialise the index of the ToA array
    i_toa = 0

    # Initialise the previous sample value
    prev_sample = arr_s[0]

    # Loop through all the samples
    for i_sample in range(1, arr_s.shape[0]):
        # Get the current sample value
        curr_sample = arr_s[i_sample]

        # Check if the threshold is crossed
        if trigger_on_rising_edge:
            if (prev_sample < threshold) and (curr_sample >= threshold):
                # Interpolate the ToA
                if i_toa >= arr_toa.shape[0]:
                    raise ValueError(
                        "The number of ToAs has exceeded the estimated number of ToAs. "
                        f"n_est is currently {n_est}. You must increase it. This error occurred"
                        f" at the sample index {i_sample} out of {arr_s.shape[0]}. "
                        f"Your ToA array has therefore been filled within {round(100 * i_sample/arr_s.shape[0])} % "
                        "of the signal."
                    )
                arr_toa[i_toa] = arr_t[i_sample - 1] + (
                    arr_t[i_sample] - arr_t[i_sample - 1]
                ) * (threshold - prev_sample) / (curr_sample - prev_sample)
                i_toa += 1
        else:
            if (prev_sample > threshold) and (curr_sample <= threshold):
                if i_toa >= arr_toa.shape[0]:
                    raise ValueError(
                        "The number of ToAs has exceeded the estimated number of ToAs. "
                        f"n_est is currently {n_est}. You must increase it. This error occurred"
                        f" at the sample index {i_sample} out of {arr_s.shape[0]}. "
                        f"Your ToA array has therefore been filled within {round(100 * i_sample/arr_s.shape[0])} % "
                        "of the signal."
                    )
                # Interpolate the ToA
                arr_toa[i_toa] = arr_t[i_sample - 1] + (
                    arr_t[i_sample] - arr_t[i_sample - 1]
                ) * (threshold - prev_sample) / (curr_sample - prev_sample)
                i_toa += 1

        # Update the previous sample value
        prev_sample = curr_sample

    # Return the array containing the ToAs
    return arr_toa[:i_toa]


@njit
def threshold_crossing_hysteresis_rising(
    arr_t: np.ndarray,
    arr_s: np.ndarray,
    threshold: float,
    hysteresis_height: float,
    n_est: Optional[float] = None,
) -> np.ndarray:
    """
    Sequentially detect rising-edge threshold crossings in a signal using hysteresis.

    A hysteresis band is established below the threshold to help
    reduce noise-induced spurious triggers.

    Parameters
    ----------
    arr_t : np.ndarray
        Time array corresponding to each sample in arr_s.
    arr_s : np.ndarray
        Signal voltage array corresponding to the time values.
    threshold : float
        Threshold value in the same units as arr_s.
    hysteresis_height : float
        Height of the hysteresis band below the threshold.
    n_est : float, optional
        The estimated number of ToAs in this
        signal. 
        Defaults to None. 
        This number is used to pre-allocate the array
        containing the ToAs. If this number is not provided, the array will
        be pre-allocated as the same dimension as arr_t and arr_s.

    Returns
    -------
    np.ndarray
        Array of interpolated rising-edge ToAs.
    """
    threshold_lower = threshold - hysteresis_height
    trigger_state = True if arr_s[0] > threshold_lower else False

    # Pre-allocate the array containing the ToAs
    if n_est is None:
        arr_toa = -1 * np.ones(arr_t.shape)
    else:
        arr_toa = -1 * np.ones(n_est)

    # Initialise the index of the ToA array
    i_toa = 0

    # Initialise the previous sample value
    prev_sample = arr_s[0]

    # Loop through all the samples
    for i_sample in range(1, arr_s.shape[0]):
        # Get the current sample value
        curr_sample = arr_s[i_sample]

        # Check if the threshold is crossed
        if trigger_state is True:
            if curr_sample <= threshold_lower:
                trigger_state = False
        else:
            if curr_sample >= threshold:
                trigger_state = True
                # Interpolate the ToA
                arr_toa[i_toa] = arr_t[i_sample - 1] + (
                    arr_t[i_sample] - arr_t[i_sample - 1]
                ) * (threshold - prev_sample) / (curr_sample - prev_sample)
                i_toa += 1

        # Update the previous sample value
        prev_sample = curr_sample

    # Return the array containing the ToAs
    return arr_toa[:i_toa]

# Untested
@njit
def threshold_crossing_hysteresis_falling(
    arr_t : np.ndarray,
    arr_s : np.ndarray,
    threshold : float,
    hysteresis_height : float,
    n_est : Optional[float] = None,
) -> np.ndarray:
    """
    Sequentially detect falling-edge threshold crossings in a signal using hysteresis.

    A hysteresis band is established above the threshold to help
    reduce noise-induced spurious triggers.

    Parameters
    ----------
    arr_t : np.ndarray
        Time array corresponding to each sample in arr_s.
    arr_s : np.ndarray
        Signal voltage array corresponding to the time values.
    threshold : float
        Threshold value in the same units as arr_s.
    hysteresis_height : float
        Height of the hysteresis band above the threshold.
    n_est : float, optional
        The estimated number of ToAs in this signal.
        Defaults to None. 
        This number is used to pre-allocate the array containing the ToAs. If this
        number is not provided, the array will be pre-allocated
        as the same dimension as arr_t and arr_s. You should
        specify this value for large signals.

    Returns
    -------
    np.ndarray
        Array of interpolated falling-edge ToAs.
    """
    threshold_upper = threshold + hysteresis_height
    trigger_state = True if arr_s[0] < threshold_upper else False

    # Pre-allocate the array containing the ToAs
    if n_est is None:
        arr_toa = -1 * np.ones(arr_t.shape)
    else:
        arr_toa = -1 * np.ones(n_est)

    # Initialise the index of the ToA array
    i_toa = 0

    # Initialise the previous sample value
    prev_sample = arr_s[0]

    # Loop through all the samples
    for i_sample in range(1, arr_s.shape[0]):
        # Get the current sample value
        curr_sample = arr_s[i_sample]

        # Check if the threshold is crossed
        if trigger_state is True:
            if curr_sample >= threshold_upper:
                trigger_state = False
        else:
            if curr_sample <= threshold:
                trigger_state = True
                # Interpolate the ToA
                arr_toa[i_toa] = (
                    arr_t[i_sample - 1] 
                    + (arr_t[i_sample] - arr_t[i_sample - 1]) 
                    * (threshold - prev_sample) 
                    / (curr_sample - prev_sample)
                )
                i_toa += 1

        # Update the previous sample value
        prev_sample = curr_sample

    # Return the array containing the ToAs
    return arr_toa[:i_toa]