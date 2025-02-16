import numpy as np
import scipy
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d

from typing import List, Optional

def lowpass_filter(signal, Wn, order=5):
    """
    Apply a low-pass filter to a signal.

    Parameters:
    signal (array-like): The input signal to be filtered.
    cutoff (float): The frequency cutoff for the low-pass filter.
    # fs (float): The sampling frequency of the signal.
    # order (int): The order of the Butterworth filter. Default is 5.
    Wn (float): The critical frequency of the filter.

    Returns:
    array-like: The filtered signal.
    """
    # Design the Butterworth low-pass filter
    # nyquist = 0.5 * fs
    # normal_cutoff = cutoff / nyquist
    normal_cutoff = Wn
    b, a = butter(order, normal_cutoff, btype='low')

    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)

    return filtered_signal

def median_savgol_filter(
    deflection_series: np.ndarray,
    kernel_size_list: List[int],
    savgol_window_length: int,
    savgol_poly_order: int,
    kernel_size_weights: Optional[List[float]] = None,
    med_savgol_weights: Optional[List[float]] = [0.5, 0.5],
    verbose: Optional[bool] = False
) -> np.ndarray:
    """
    Filters the deflection series using a combination of median and Savitzky-Golay filters.

    Parameters
    ----------
    deflection_series : np.ndarray
        The deflection series to be filtered.
    kernel_size_list : List[int]
        The kernel sizes to use for the median filter.
    savgol_window_length : int
        The length of the window to use for the Savitzky-Golay filter.
    savgol_poly_order : int
        The polynomial order for the Savitzky-Golay filter.
    kernel_size_weights : Optional[List[float]], default=None
        The weights to use for the kernel sizes. If None, equal weights are used.
    med_savgol_weights : Optional[List[float]], default=[0.5, 0.5]
        The weights to use for combining the median and Savitzky-Golay filters.
    verbose : Optional[bool], default=False
        A flag to enable verbose output.

    Returns
    -------
    np.ndarray
        The filtered deflection series.
    """
    for kernel_counter, kernel_size_i_test in enumerate(kernel_size_list):
        if kernel_size_i_test % 2 == 0:
            kernel_size_list[kernel_counter] = kernel_size_i_test + 1
            if verbose:
                print(f"Adding 1 to index {kernel_counter} in kernel_size_list (old value: {kernel_size_i_test}, updated value: {kernel_size_list[kernel_counter]})")

    filt_i = np.ones(shape=(len(kernel_size_list), len(deflection_series)))

    if kernel_size_weights is None:
        kernel_size_weights = np.ones(len(kernel_size_list))

    if len(kernel_size_weights) != len(kernel_size_list):
        raise ValueError(f"The length of the kernel_size_weights list ({len(kernel_size_weights)}) must match the length of the kernel_size_list list ({len(kernel_size_list)}).")

    for i, kernel_size_i in enumerate(kernel_size_list):
        filt_i[i, :] = scipy.signal.medfilt(deflection_series, kernel_size=kernel_size_i)

    filt_kernel_sizes_mean = np.average(filt_i, axis=0, weights=kernel_size_weights)
    savgol_filt = scipy.signal.savgol_filter(deflection_series, savgol_window_length, savgol_poly_order, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0)

    filtered_series = np.average(np.vstack((filt_kernel_sizes_mean, savgol_filt)), axis=0, weights=med_savgol_weights)
    return filtered_series


def gaussian_filter(deflection_series: np.ndarray, sigma: float, order: int) -> np.ndarray:
    """
    Applies a Gaussian filter to the deflection series.

    Parameters
    ----------
    deflection_series : np.ndarray
        The deflection series to be filtered.
    sigma : float
        The standard deviation for Gaussian kernel.

    Returns
    -------
    np.ndarray
        The filtered deflection series.
    """
    return gaussian_filter1d(deflection_series, sigma=sigma, order=order)

# Still add hankel-ICA and hankel-PCA filter, just need to sort out references for Jesse Stevens' functions and code