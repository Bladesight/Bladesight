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
    """A sequential threshold crossing algorithm that interpolates
        the ToA between the two samples where the signal crosses
        the threshold.

    Args:
        arr_t (np.ndarray): The array containing the time values.
        arr_s (np.ndarray): The array containing the signal voltage values
            corresponding to the time values.
        threshold (float): The threshold value.
        n_est (float, optional): The estimated number of ToAs in this signal. Defaults
            to None. This number is used to pre-allocate the array containing the
            ToAs. If this number is not provided, the array will be pre-allocated as
            the same dimension as arr_t and arr_s.
        trigger_on_rising_edge (bool, optional): Whether to trigger ToAs on the rising
            or falling edge. Defaults to True. If True, the ToA is triggered on
            the rising edge.

    Returns:
        np.ndarray: An array containing the ToAs.
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
    """A sequential threshold crossing algorithm that interpolates
        the ToA between the two samples where the signal crosses
        the threshold.

    Args:
        arr_t (np.ndarray): The array containing the time values.
        arr_s (np.ndarray): The array containing the signal voltage values
            corresponding to the time values.
        threshold (float): The threshold value.
        hysteresis_height (float): The height of the hysteresis, in the same
            units as the signal.
        n_est (float, optional): The estimated number of ToAs in this
            signal. Defaults to None. This number is used to pre-allocate the array
            containing the ToAs. If this number is not provided, the array will
            be pre-allocated as the same dimension as arr_t and arr_s.

    Returns:
        np.ndarray: An array containing the ToAs.
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
    """ This function implements the constant threshold triggering
    method with hysteresis on the falling edge. The hysteresis
    height is specified in the same units as the signal.

    Args:
        arr_t (np.ndarray): The time values of the signal.
        arr_s (np.ndarray): The signal to determine the threshold
            level for.
        threshold (float): The threshold level to use for the
            constant threshold triggering method.
        hysteresis_height (float): The height of the hysteresis.
            It has the same units as the signal.
        n_est (Optional[float]): The estimated number of ToAs in
            this signal. Defaults to None. This number is used to
            pre-allocate the array containing the ToAs. If this
            number is not provided, the array will be pre-allocated
            as the same dimension as arr_t and arr_s. You should
            specify this value for large signals.
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