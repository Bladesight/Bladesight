from typing import Optional, Tuple

import numpy as np
from numba import njit

from bladesight.btt import verbose_print

def get_constant_thresholds(
    sensor_array: np.ndarray,
    sensor_type: str,
    threshold_hysteresis_dict: dict[str, float],
    threshold_category: Optional[str] = "correct",
    verbose: Optional[bool] = False,
    ) -> Tuple[float, float]:
    """
    Determine the threshold level to use for the constant threshold trigger method.

    Parameters
    ----------
    sensor_array : np.ndarray
        The time signal to determine the threshold level for.
    sensor_type : str
        The type of sensor.
    threshold_hysteresis_dict : dict[str, float]
        The dictionary of threshold levels for each category. The keys must be in the format
        "<sensor type> <"threshold" or "hysteresis" keyword> <level>". The level is a keyword
        between "low", "correct", or "high" that can be used for different triggering threshold studies.
        The float associated with each key is the threshold or hysteresis level expressed as a percentage value.
        Note that if no "hysteresis" keyword is found in the dictionary, the hysteresis level is set to None.
    threshold_category : Optional[str], optional
        The category of threshold level to use ('low', 'correct', or 'high'), by default 'correct'.

    Returns
    -------
    Tuple[float, float]
        The threshold voltage level and hysteresis voltage level.

    Raises
    ------
    ValueError
        If threshold_category is not one of the following: "low", "correct", "high".
    ValueError
        If threshold_category_array contains values outside the range of 0 to 100.

    Notes:
    ------
    This function is modified/developed by Justin Smith using existing bladesight code (see References).    

    Example Usage:
    --------------


    tacho_threshold_hysteresis_category_dict = {
            'tacho OPR threshold correct': 60,
            'tacho OPR hysteresis correct': 10,
            'tacho MPR threshold correct': 60,
            'tacho MPR hysteresis correct': 55,
            }
    sensor_type = "Tacho OPR"
    OPR_signal = np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1, 0.7, 0.3, 0, 0, 0])
    threshold_OPR, hysteresis_OPR = get_constant_thresholds(OPR_signal, sensor_type = sensor_type, threshold_hysteresis_dict = tacho_threshold_hysteresis_category_dict)

    >>>>(0.6, 0.06)

    References
    ----------
    This function is adapted from determine_threshold_level in Chapter 2 of the bladesight tutorial
    (https://docs.bladesight.com/tutorials/intro_to_btt/ch2/#problem-1-automatic-range-detection).
    [1] D. H. Diamond, “Introduction to Blade Tip Timing,” Bladesight Learn. Accessed: Feb. 12, 2024.
    [Online]. Available: docs.bladesight.com
    """
    # Checking theshold_category and threshold values are appropriate
    if threshold_category not in ["low", "correct", "high"]:
        raise ValueError(
            "threshold_category must be one of the following: 'low', 'correct', 'high'"
        )
    if (
        any(value < 0 for value in threshold_hysteresis_dict.values()) < 0
        or any(value < 0 for value in threshold_hysteresis_dict.values()) > 100
    ):
        raise ValueError("threshold_category_array must be between 0 and 100")
    
    sensor_threshold_keys = [
    key for key in threshold_hysteresis_dict.keys() if key.lower().startswith(sensor_type.lower())
    ]

    if not sensor_threshold_keys:
        avail_keys = f"Available Keys: {threshold_hysteresis_dict.keys()}"
        raise ValueError(f"{avail_keys}\n" + f"No keys found in threshold_hysteresis_dict for sensor type '{sensor_type}'")


    threshold_keys = [key for key in sensor_threshold_keys if "threshold" in key]

    if not threshold_keys:
        avail_keys = f"Available Keys: {threshold_hysteresis_dict.keys()}"
        raise ValueError(f"{avail_keys}\n" + f"No threshold keys found in threshold_hysteresis_dict for sensor type '{sensor_type}'")

    sensor_hysteresis_keys = [
        key for key in threshold_hysteresis_dict.keys() if key.lower().startswith(sensor_type.lower())
    ]


    hysteresis_keys = [key for key in sensor_hysteresis_keys if "hysteresis" in key.lower()]
    if not hysteresis_keys:
        avail_keys = f"Available Keys: {threshold_hysteresis_dict.keys()}"
        raise ValueError(f"{avail_keys}\n" + f"No hysteresis keys found in threshold_hysteresis_dict for sensor type '{sensor_type}'")

    threshold_level_key = [key for key in threshold_keys if threshold_category.lower() in key.lower()]
    if not threshold_level_key:
        raise ValueError(f"No threshold level key found for category '{threshold_category}' in threshold_hysteresis_dict for sensor type '{sensor_type}'")


    threshold_level_percent = threshold_hysteresis_dict[threshold_level_key[0]]

    if (not hysteresis_keys) != True: #if hysteresis_keys is not empty, then dont assign a hysteresis level
        hysteresis_level_key = [key for key in hysteresis_keys if threshold_category.lower() in key.lower()]
        hysteresis_level_percent = threshold_hysteresis_dict[hysteresis_level_key[0]]

    if (not hysteresis_keys) != False:
        hysteresis_level_key = None
        hysteresis_level_percent = None

    min_value = np.min(sensor_array)
    max_value = np.max(sensor_array)
    signal_range = max_value - min_value

    verbose_print(verbose, f"sensor_type: {sensor_type}")
    verbose_print(verbose, f"Min Value: {min_value} Volts")
    verbose_print(verbose, f"Max Value: {max_value} Volts")
    # verbose_print(verbose, f"mean value: {np.mean(sensor_array)} Volts")
    # verbose_print(verbose, f"10th percentile value: {np.percentile(sensor_array, 10)} Volts")
    verbose_print(verbose, f"Signal Range: {signal_range} Volts")

    threshold_level = min_value + signal_range * threshold_level_percent / 100 # Volts

    if hysteresis_level_percent is None:
        hysteresis_level = None
        
    if hysteresis_level_percent is not None:
        hysteresis_level = threshold_level * hysteresis_level_percent / 100 # Volts

    verbose_print(verbose, f"Sensor Type: {sensor_type}")
    verbose_print(verbose, f"THRESHOLD_LEVEL: {threshold_level_percent}% => {threshold_level} Volts")
    verbose_print(verbose, f"HYSTERESIS_LEVEL: {hysteresis_level_percent}% => {hysteresis_level} Volts")

    return threshold_level, hysteresis_level

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