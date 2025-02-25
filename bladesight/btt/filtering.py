import numpy as np
import scipy
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from typing import List, Optional, Tuple, Callable


def lowpass_filter(signal: np.ndarray, Wn: float, order: int = 5) -> np.ndarray:
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
    b, a = butter(order, normal_cutoff, btype="low")

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
    verbose: Optional[bool] = False,
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
                print(
                    f"Adding 1 to index {kernel_counter} in kernel_size_list (old value: {kernel_size_i_test}, updated value: {kernel_size_list[kernel_counter]})"
                )

    filt_i = np.ones(shape=(len(kernel_size_list), len(deflection_series)))

    if kernel_size_weights is None:
        kernel_size_weights = np.ones(len(kernel_size_list))

    if len(kernel_size_weights) != len(kernel_size_list):
        raise ValueError(
            f"The length of the kernel_size_weights list ({len(kernel_size_weights)}) must match the length of the kernel_size_list list ({len(kernel_size_list)})."
        )

    for i, kernel_size_i in enumerate(kernel_size_list):
        filt_i[i, :] = scipy.signal.medfilt(
            deflection_series, kernel_size=kernel_size_i
        )

    filt_kernel_sizes_mean = np.average(filt_i, axis=0, weights=kernel_size_weights)
    savgol_filt = scipy.signal.savgol_filter(
        deflection_series,
        savgol_window_length,
        savgol_poly_order,
        deriv=0,
        delta=1.0,
        axis=-1,
        mode="interp",
        cval=0.0,
    )

    filtered_series = np.average(
        np.vstack((filt_kernel_sizes_mean, savgol_filt)),
        axis=0,
        weights=med_savgol_weights,
    )
    return filtered_series


def gaussian_filter(
    deflection_series: np.ndarray, sigma: float, order: int
) -> np.ndarray:
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


def apply_PCA(
    hankel_matrix: np.ndarray, n_components: int, plot_components=False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply PCA to the Hankel matrix and reconstruct it.

    Parameters
    ----------
    hankel_matrix : np.ndarray
        The standardized Hankel matrix.
    n_components : int
        The number of principal components to keep.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        The principal components, the reconstructed Hankel matrix, and the explained variance ratio.
    """
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(hankel_matrix)
    reconstructed_hankel = pca.inverse_transform(principal_components)
    explained_variance_ratio = pca.explained_variance_ratio_
    print(
        "Variance explained by PCA components:",
        explained_variance_ratio,
        "sum:",
        sum(explained_variance_ratio),
    )

    if plot_components == True:
        plt.figure()
        plt.title("PCA Component Scores")
        # Plot component scores
        ax1 = plt.gca()
        (line1,) = ax1.plot(
            np.arange(1, n_components + 1, 1),
            explained_variance_ratio,
            marker="o",
            linestyle="-",
            label=f"Component Scores, sum = {np.round(sum(explained_variance_ratio), 2)}",
        )
        ax1.set_xlabel("Component Index")
        ax1.set_ylabel("Score")
        # Create a second y-axis for cumulative scores
        ax2 = ax1.twinx()
        (line2,) = ax2.plot(
            np.arange(1, n_components + 1, 1),
            np.cumsum(explained_variance_ratio),
            marker="x",
            linestyle="--",
            color="r",
            label="Cumulative Score",
        )
        ax2.set_ylabel("Cumulative Score")
        # Combine legends from both y-axes
        lines = [line1, line2]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc="best")
        plt.show()

    return principal_components, reconstructed_hankel, explained_variance_ratio


def hankel_denoising(
    signal: np.ndarray,
    n_components: int = 1,
    hankel_size: int = 10,
    decomposition_function: Callable[
        [np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = apply_PCA,
    decomposition_function_args: Optional[Tuple] = None,
    scaler_preprocessing_instance: Optional[StandardScaler] = StandardScaler(with_mean = True, with_std = True),
    # scaler_preprocessing_dict: Optional[dict] = {'with_mean': True, 'with_std': True},
    scaler_hankel_instance: Optional[StandardScaler] = StandardScaler(with_mean = True, with_std = True),
) -> np.ndarray:
    """
    Denoise a signal using Hankel matrix and a decomposition method (PCA or ICA).

    Parameters
    ----------
    signal : np.ndarray
        The input signal to be denoised.
    n_components : int, optional
        The number of components to keep, by default 1.
    hankel_size : int, optional
        The size of the Hankel matrix, by default 10.
    decomposition_function : Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]], optional
        The decomposition function to apply (e.g., apply_PCA or apply_ICA), by default apply_PCA.

    Returns
    -------
    np.ndarray
        The denoised signal.
    """

    # Standardising the data before performing PCA or ICA
    # signal_scalar = StandardScaler(with_mean=True, with_std=True)
    # signal = signal_scalar.fit_transform(signal.reshape(-1, 1)).reshape(-1)
    if scaler_preprocessing_instance is not None:
        signal = scaler_preprocessing_instance.fit_transform(signal.reshape(-1, 1)).reshape(-1)
    if scaler_preprocessing_instance is None:
        signal = signal

    signal = np.asarray(signal)  # Ensure the signal is a NumPy array
    N = len(signal)

    # Create the Hankel matrix
    hankel_matrix = hankel(signal[:hankel_size], signal[hankel_size - 1 :])
    # print("hankel_matrix.shape:", hankel_matrix.shape)

    # Standardize the Hankel matrix
    # scaler = StandardScaler()
    # hankel_standardized = scaler.fit_transform(hankel_matrix)
    if scaler_hankel_instance is not None:
        hankel_standardized = scaler_hankel_instance.fit_transform(hankel_matrix)
    if scaler_hankel_instance is None:
        hankel_standardized = hankel_matrix

    # Apply the decomposition function (PCA or ICA)
    if decomposition_function_args is None:
        _, reconstructed_hankel, _ = decomposition_function(
            hankel_standardized, n_components
        )
    else:
        _, reconstructed_hankel, _ = decomposition_function(
            hankel_standardized,
            n_components=n_components,
            **decomposition_function_args,
        )

    # Inverse transform to original space
    if scaler_hankel_instance is not None:
        denoised_hankel = scaler_hankel_instance.inverse_transform(reconstructed_hankel)
    if scaler_hankel_instance is None:
        denoised_hankel = reconstructed_hankel

    # Average anti-diagonals to reconstruct the 1D signal
    denoised_signal = np.array(
        [
            np.mean(np.diag(denoised_hankel[:, ::-1], k))
            for k in range(-hankel_size + 1, N - hankel_size + 1)
        ]
    )

    # Align the reconstructed signal with the original signal
    cross_correlation = np.correlate(signal, denoised_signal, mode="full")
    lag = np.argmax(cross_correlation) - (len(denoised_signal) - 1)
    denoised_signal_aligned = np.roll(denoised_signal, lag)

    if scaler_preprocessing_instance is not None: 
        denoised_signal_aligned = scaler_preprocessing_instance.inverse_transform(
            denoised_signal_aligned.reshape(-1, 1)
        ).reshape(-1)
    if scaler_preprocessing_instance is None:
        denoised_signal_aligned = denoised_signal_aligned.reshape(-1)

    return denoised_signal_aligned
