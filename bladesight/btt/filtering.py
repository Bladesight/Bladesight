import numpy as np
import scipy
import scipy.stats
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from scipy.linalg import hankel
from sklearn.decomposition import PCA, FastICA
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


class ICA_ranker:
    """
    Ranks independent components based on their similarity to the original signal.
    
    This class takes independent components, the original signal, and a metric function,
    then computes scores for each component and ranks them in descending order of importance.
    
    Parameters
    ----------
    components : np.ndarray
        The independent components to rank, shape (n_components, n_samples).
    signal : np.ndarray
        The original signal or reference for comparison.
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Function that compares two signals and returns a similarity score.
        Default is innerProduct.
        
    Attributes
    ----------
    components : np.ndarray
        The input components.
    signal : np.ndarray
        The input signal used for comparison.
    metric : Callable
        The metric function used for comparison.
    sorts : np.ndarray
        The indices that would sort the components by their scores.
    scores : np.ndarray
        The normalized scores (0-100) for each component, sorted in descending order.
    rankedComponents : np.ndarray
        The components sorted by their scores in descending order.
        
    Notes
    -----
    The ranking process:
    1. Compute metric value between each component and the original signal
    2. Normalize the absolute metric values to sum to 1
    3. Convert to percentages (0-100)
    4. Sort components by their scores in descending order
        
    References
    ----------
    .. [1] J. Stevens, D. N. Wilke, and I. I. Setshedi, "Enhancing LS-PIE's Optimal 
            Latent Dimensional Identification: Latent Expansion and Latent Condensation",
            MCA, vol. 29, no. 4, p. 65, Aug. 2024, doi: 10.3390/mca29040065.
    ..      Please see the following GitHub for the corresponding code: https://github.com/Greeen16/LS-PIE
    """
    @staticmethod
    def innerProduct(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute inner product between two signals.
        
        Parameters
        ----------
        x : np.ndarray
            First signal.
        y : np.ndarray
            Second signal.
            
        Returns
        -------
        float
            Sum of the dot product between the two signals.
        """
        return np.sum(np.dot(y, x))
    @staticmethod
    def kurtosis(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the difference in kurtosis between two signals.
        
        Parameters
        ----------
        x : np.ndarray
            First signal.
        y : np.ndarray
            Second signal.
            
        Returns
        -------
        float
            Sum of the difference in kurtosis between the two signals.
        """
        return np.sum(scipy.stats.kurtosis(x) - scipy.stats.kurtosis(y))# + np.sum(np.dot(y, x))
    
    def __init__(self, components: np.ndarray, signal: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float] = innerProduct):
        self.components = components
        self.signal = signal
        self.metric = metric

        # Collect the scores
        metrics = [metric(signal, component) for component in components]
        # print("metrics = ", metrics)
        
        # Normalize and sort
        tot = np.sum([abs(met) for met in metrics])
        Metrics = np.array([abs(met) for met in metrics]) / tot
        sorts = Metrics.argsort()[::-1]
        
        self.sorts = sorts
        self.scores = Metrics[sorts] * 100
        self.rankedComponents = np.array(components)[sorts]

class ICA_scaler:
    """
    Scales independent components based on their importance to the original signal.
    
    This class ranks components using ICA_ranker, then scales each component 
    by its normalized score.
    
    Parameters
    ----------
    components : np.ndarray
        The independent components to scale, shape (n_components, n_samples).
    signal : np.ndarray
        The original signal or reference for comparison.
    metric : Callable[[np.ndarray, np.ndarray], float], optional
        Function that compares two signals and returns a similarity score.
        Default is innerProduct.
        
    Attributes
    ----------
    components : np.ndarray
        The input components.
    signal : np.ndarray
        The input signal used for comparison.
    metric : Callable
        The metric function used for comparison.
    scores : np.ndarray
        The normalized scores (0-100) for each component, sorted in descending order.
    scaledComponents : list
        Components scaled by their scores, where each component is multiplied
        by its normalized score (score/100).
        
    Notes
    -----
    The scaling process:
    1. Rank components using ICA_ranker
    2. Scale each ranked component by its normalized score (score/100)
    
    References
    ----------
    .. [1] J. Stevens, D. N. Wilke, and I. I. Setshedi, "Enhancing LS-PIE's Optimal 
            Latent Dimensional Identification: Latent Expansion and Latent Condensation",
            MCA, vol. 29, no. 4, p. 65, Aug. 2024, doi: 10.3390/mca29040065.
    ..      Please see the following GitHub for the corresponding code: https://github.com/Greeen16/LS-PIE
    """
    @staticmethod
    def innerProduct(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute inner product between two signals.
        
        Parameters
        ----------
        x : np.ndarray
            First signal.
        y : np.ndarray
            Second signal.
            
        Returns
        -------
        float
            Sum of the dot product between the two signals.
        """
        return np.sum(np.dot(y, x))
    @staticmethod
    def kurtosis(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the difference in kurtosis between two signals.
        
        Parameters
        ----------
        x : np.ndarray
            First signal.
        y : np.ndarray
            Second signal.
            
        Returns
        -------
        float
            Sum of the difference in kurtosis between the two signals.
        """
        return np.sum(scipy.stats.kurtosis(x) - scipy.stats.kurtosis(y))# + np.sum(np.dot(y, x))
        
    def __init__(self, components: np.ndarray, signal: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float] = innerProduct):
        self.components = components
        self.signal = signal
        self.metric = metric

        ranking = ICA_ranker(self.components, self.signal, self.metric)
        self.scores = ranking.scores
        rankedComps = ranking.rankedComponents
        self.scaledComponents = [rankedComps[i] * self.scores[i] / 100 for i in range(len(self.scores))]
        # print("self.scaledComponents:", self.scaledComponents)

def apply_ICA(hankel_matrix: np.ndarray, n_components: int, n_reconstruction_components: int = 3, plot_components = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Independent Component Analysis (ICA) to a Hankel matrix and reconstruct it.
    
    This function applies FastICA to extract independent components from the Hankel matrix,
    then ranks and scales these components to reconstruct a denoised version of the matrix.
    
    Parameters
    ----------
    hankel_matrix : np.ndarray
        The standardized Hankel matrix of shape (rows, cols).
    n_components : int
        The number of independent components to extract from the matrix.
    n_reconstruction_components : int, optional
        The number of components to use for reconstruction. Must be ≤ n_components.
        Default is 3.
    plot_components : bool, optional
        Whether to generate a plot showing component scores and cumulative scores.
        Default is False.
        
    Returns
    -------
    independent_components : np.ndarray
        The extracted independent components, shape (n_components, n_samples).
    reconstructed_hankel : np.ndarray
        The reconstructed Hankel matrix using the top n_reconstruction_components.
    mixing_matrix : np.ndarray
        The mixing matrix from the FastICA algorithm.
        
    Notes
    -----
    The reconstruction process:
    1. Apply FastICA to extract independent components
    2. Rank and scale components by importance using ICA_scaler
    3. Use the top n_reconstruction_components to reconstruct the Hankel matrix
    4. The reconstructed matrix is calculated as: 
        scaled_components.T @ mixing_matrix.T[:n_reconstruction_components, :]
    
    References
    ----------
    .. [1] J. Stevens, D. N. Wilke, and I. I. Setshedi, "Enhancing LS-PIE's Optimal
            Latent Dimensional Identification: Latent Expansion and Latent Condensation",
            MCA, vol. 29, no. 4, p. 65, Aug. 2024, doi: 10.3390/mca29040065.
    ..      Please see the following GitHub for the corresponding code:
    ..

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import hankel
    >>> # Create a test signal with noise
    >>> t = np.linspace(0, 10, 1000)
    >>> signal = np.sin(t) + 0.2 * np.random.randn(len(t))
    >>> # Create Hankel matrix
    >>> h_size = 20
    >>> h_matrix = hankel(signal[:h_size], signal[h_size-1:])
    >>> # Apply ICA
    >>> _, reconstructed, _ = apply_ICA(h_matrix, n_components=10, 
    ...                                n_reconstruction_components=3)
    >>> print(f"Original shape: {h_matrix.shape}, Reconstructed: {reconstructed.shape}")
    """
    ica = FastICA(n_components=n_components, tol = 1E-6)#, whiten=True) # Use whiten=False as i think sklearn is pre-processing the data and that is effecting reconstruction performance 
    independent_components = ica.fit_transform(hankel_matrix).T
    # print("independent_components:", independent_components)
    # print("independent_components.shape:", independent_components.shape)
    mixing_matrix = ica.mixing_
    # print("Mixing matrix.shape:", mixing_matrix.shape)
    
    # Rank and scale the components
    scaler_instance = ICA_scaler(independent_components, hankel_matrix)
    # print("scaler_instance.scaledComponents:", scaler_instance.scaledComponents)
    # scaled_components = np.array(scaler_instance.scaledComponents[:n_reconstruction_components])  # Us only the specified number of components
    scaled_components = np.array(scaler_instance.scaledComponents[:n_reconstruction_components])  # Use only the specified number of components
    # print('scaled_components:', scaled_components)
    # print('scaled_components:', scaled_components.shape)
    if plot_components == True:
        plt.figure()
        plt.title('ICA Component Scores')
        # Plot component scores
        ax1 = plt.gca()
        line1, = ax1.plot(np.arange(1, n_components + 1, 1), scaler_instance.scores, marker='o', linestyle='-', label=f'Component Scores, sum = {np.round(sum(scaler_instance.scores), 2)}')
        line2, = ax1.plot(np.arange(1, n_reconstruction_components + 1, 1), scaler_instance.scores[:n_reconstruction_components], marker='.', linestyle='-', label=f'First {n_reconstruction_components} Components, sum = {np.round(sum(scaler_instance.scores[:n_reconstruction_components]), 2)}')
        ax1.set_xlabel('Component Index')
        ax1.set_ylabel('Score')
        # Create a second y-axis for cumulative scores
        ax2 = ax1.twinx()
        line3, = ax2.plot(np.arange(1, n_components + 1, 1), np.cumsum(scaler_instance.scores), marker='x', linestyle='--', color='r', label='Cumulative Score')
        ax2.set_ylabel('Cumulative Score')
        # Combine legends from both y-axes
        lines = [line1, line2, line3]
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, loc='best')
        plt.show()

    reconstructed_hankel = np.dot(scaled_components.T, mixing_matrix.T[:n_reconstruction_components, :])
    # reconstructed_hankel = np.dot(scaled_components, mixing_matrix)#[:n_reconstruction_components, :])
    
    return independent_components, reconstructed_hankel, mixing_matrix

def hankel_denoising(
    signal: np.ndarray,
    n_components: int = 1,
    hankel_size: int = 10,
    decomposition_function: Callable[
        [np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ] = apply_PCA,
    decomposition_function_args: Optional[dict] = None,
    scaler_preprocessing_instance: Optional[StandardScaler] = StandardScaler(with_mean=True, with_std=True),
    scaler_hankel_instance: Optional[StandardScaler] = StandardScaler(with_mean=True, with_std=True),
    pre_filter_function: Optional[Callable] = None,
    pre_filter_args: Optional[dict] = {},
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
    decomposition_function_args : Optional[dict], optional
        The arguments to pass to the decomposition function, by default None.
    scaler_preprocessing_instance : Optional[StandardScaler], optional
        The instance of the StandardScaler to use for preprocessing the signal, by default StandardScaler(with_mean=True, with_std=True).
    scaler_hankel_instance : Optional[StandardScaler], optional
        The instance of the StandardScaler to use for preprocessing the Hankel matrix, by default StandardScaler(with_mean=True, with_std=True).
    pre_filter_function : Optional[Callable], optional
        Function to pre-filter the signal before any other processing (e.g., lowpass_filter), by default None.
    pre_filter_args : Optional[Dict], optional
        Arguments to pass to the pre_filter_function, by default {}. These arguments should match the parameters expected by the pre_filter_function.

    Returns
    -------
    np.ndarray
        The denoised signal.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.preprocessing import StandardScaler
    >>> 
    >>> def lowpass_filter(signal, Wn, order=5):
    ...     from scipy.signal import butter, filtfilt
    ...     b, a = butter(order, Wn, btype='low')
    ...     return filtfilt(b, a, signal)
    >>> 
    >>> signal = np.random.randn(100)
    >>> denoised_signal = hankel_denoising(
    ...     signal,
    ...     n_components=2,
    ...     hankel_size=10,
    ...     decomposition_function=apply_PCA,
    ...     pre_filter_function=lowpass_filter,
    ...     pre_filter_args={'Wn': 0.1, 'order': 3}
    ... )
    >>> print(denoised_signal)
    """
    # Ensure the signal is a NumPy array
    signal = np.asarray(signal)
    original_signal = signal.copy()  # Keep original for alignment later
    
    # Apply pre-filtering if specified
    if pre_filter_function is not None:
        if pre_filter_args is None:
            pre_filter_args = {}
        signal = pre_filter_function(signal, **pre_filter_args)
    
    # Standardizing the data before performing PCA or ICA
    if scaler_preprocessing_instance is not None:
        signal = scaler_preprocessing_instance.fit_transform(signal.reshape(-1, 1)).reshape(-1)

    N = len(signal)

    # Create the Hankel matrix
    hankel_matrix = hankel(signal[:hankel_size], signal[hankel_size - 1:])

    # Standardize the Hankel matrix
    if scaler_hankel_instance is not None:
        hankel_standardized = scaler_hankel_instance.fit_transform(hankel_matrix)
    else:
        hankel_standardized = hankel_matrix

    # Apply the decomposition function (PCA or ICA)
    if decomposition_function_args is None:
        decomposition_function_args = {}
    
    _, reconstructed_hankel, _ = decomposition_function(
        hankel_standardized, n_components, **decomposition_function_args
    )

    # Inverse transform to original space
    if scaler_hankel_instance is not None:
        denoised_hankel = scaler_hankel_instance.inverse_transform(reconstructed_hankel)
    else:
        denoised_hankel = reconstructed_hankel

    # Average anti-diagonals to reconstruct the 1D signal
    denoised_signal = np.array(
        [
            np.mean(np.diag(denoised_hankel[:, ::-1], k))
            for k in range(-hankel_size + 1, N - hankel_size + 1)
        ]
    )

    # Align the reconstructed signal with the original signal (before filtering)
    # This ensures better alignment when pre-filtering might introduce phase shifts
    cross_correlation = np.correlate(original_signal, denoised_signal, mode="full")
    lag = np.argmax(cross_correlation) - (len(denoised_signal) - 1)
    denoised_signal_aligned = np.roll(denoised_signal, lag)

    # Inverse transform if we used preprocessing scaling
    if scaler_preprocessing_instance is not None:
        denoised_signal_aligned = scaler_preprocessing_instance.inverse_transform(
            denoised_signal_aligned.reshape(-1, 1)
        ).reshape(-1)

    return denoised_signal_aligned

def hankel_denoising_2D(
    signals: np.ndarray,  # shape (n_signals, n_samples)
    n_components: int,
    hankel_size: int,
    decomposition_function: Callable[
        [np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ],  # e.g., apply_PCA
    decomposition_function_args: Optional[dict] = None,
    scaler_preprocessing_instance: Optional[StandardScaler] = StandardScaler(with_mean=True, with_std=True),
    scaler_hankel_instance: Optional[StandardScaler] = StandardScaler(with_mean=True, with_std=True),
    target_signal_index: int = 0,
) -> np.ndarray:
    """
    Denoise multiple signals using a Hankel transform and a decomposition method (e.g., PCA).
    Each row of 'signals' is treated as a separate 1D signal. After decomposition and reconstruction,
    only the denoised version of the target signal (specified by 'target_signal_index') is returned.

    Parameters
    ----------
    signals : np.ndarray
        A 2D array of shape (n_signals, n_samples), where each row is one signal.
    n_components : int, optional
        The number of components to keep, by default 1.
    hankel_size : int, optional
        The size of the Hankel matrix, by default 10.
    decomposition_function : Callable[[np.ndarray, int], Tuple[np.ndarray, np.ndarray, np.ndarray]]
        A decomposition function (e.g., PCA or ICA) that takes a 2D array and returns
        (principal_components, reconstructed, explained_variance_ratio).
    decomposition_function_args : Optional[dict], optional
        Arguments to pass to the decomposition function, by default None.
    scaler_preprocessing_instance : Optional[StandardScaler], optional
        StandardScaler for preprocessing each signal before Hankel transform, by default StandardScaler().
    scaler_hankel_instance : Optional[StandardScaler], optional
        StandardScaler for preprocessing the stacked Hankel matrix, by default StandardScaler().
    target_signal_index : int, optional
        Which row in 'signals' to return after denoising. Default is 0.

    Returns
    -------
    np.ndarray
        The denoised (reconstructed) 1D signal corresponding to 'target_signal_index'.

    Notes
    -----
    - Each signal is converted to a Hankel matrix of shape (hankel_size, n_cols), then
        all Hankel matrices are stacked along the row dimension.
    - The decomposition is applied once to the stacked matrix, and each Hankel matrix
        is reconstructed. Finally, the target signal is extracted via anti-diagonal averaging
        and optionally realigned by cross-correlation.
    """

    # Ensure signals is at least 2D (n_signals, n_samples)
    signals = np.atleast_2d(signals)
    print("signals.shape:", signals.shape)
    n_signals, n_samples = signals.shape
    print(f"{n_signals=}, {n_samples=}")
    # Preprocess each signal individually if scaler_preprocessing_instance is provided
    # ------OLD CODE------
    if scaler_preprocessing_instance is not None:
        for i in range(n_signals):
            signals[i, :] = scaler_preprocessing_instance.fit_transform(
                signals[i, :].reshape(-1, 1)
            ).ravel()
    # ------END OLD CODE------
    # if scaler_preprocessing_instance is not None:
    #     for i in range(n_signals):
    #         signals[i, :] = scaler_preprocessing_instance.fit_transform(
    #             signals[i, :].reshape(-1, 1)
    #         ).reshape(-1, n_samples)
    #     print("standardised signals.shape:", signals.shape)
    print("signals preprocessed.shape:", signals.shape)
    
    # Build Hankel matrices for each signal and stack them vertically
    hankel_list = []
    print("hankel_size", hankel_size)
    for i in range(n_signals):
        # Create Hankel for signal i
        hankel_matrix = hankel(signals[i, :hankel_size], signals[i, hankel_size - 1 :])
        # hankel_matrix = hankel(signals[:hankel_size, i], signals[hankel_size - 1 :, i])
        hankel_list.append(hankel_matrix)
    print('hankel_matrix.shape', hankel_matrix.shape)

    # Stacked shape: (n_signals * hankel_size, n_cols)
    stacked_hankel = np.vstack(hankel_list)
    print('stacked_hankel.shape:', stacked_hankel.shape)
    
    # Optionally scale the stacked Hankel matrix
    if scaler_hankel_instance is not None:
        stacked_hankel = scaler_hankel_instance.fit_transform(stacked_hankel)

    if decomposition_function_args is None:
        decomposition_function_args = {}

    _, reconstructed, _ = decomposition_function(
        stacked_hankel, n_components, **decomposition_function_args
    )

    # Inverse transform the Hankel if it was scaled
    if scaler_hankel_instance is not None:
        reconstructed = scaler_hankel_instance.inverse_transform(reconstructed)

    # Re-split the reconstructed Hankel rows into separate signals
    denoised_hankels = []
    rows_per_signal = hankel_size
    n_cols = reconstructed.shape[1]

    start_row = 0
    for _ in range(n_signals):
        end_row = start_row + rows_per_signal
        denoised_hankels.append(reconstructed[start_row:end_row, :])
        start_row = end_row

    # Anti-diagonal averaging to convert each Hankel matrix back to a 1D signal
    denoised_signals = []
    for i in range(n_signals):
        hmat = denoised_hankels[i]
        # Each step k is a diagonal in the reversed columns
        # Range ensures the correct length matches the original signal
        dsig = np.array([
            np.mean(np.diag(hmat[:, ::-1], k))
            for k in range(-hankel_size + 1, n_samples - hankel_size + 1)
        ])
        # Align with the original signal
        cross_correlation = np.correlate(signals[i, :], dsig, mode="full")
        lag = np.argmax(cross_correlation) - (len(dsig) - 1)
        dsig_aligned = np.roll(dsig, lag)
        denoised_signals.append(dsig_aligned)

    # Ensure the denoised signals have the same length as the input signals
    for i in range(n_signals):
        if len(denoised_signals[i]) < n_samples:
            padding = np.zeros(n_samples - len(denoised_signals[i]))
            denoised_signals[i] = np.concatenate((denoised_signals[i], padding))

    # Reverse any initial per-signal preprocessing
    if scaler_preprocessing_instance is not None:
        # We need a consistent inverse_transform, so re-fit each row individually
        # or store the fitted scalers earlier. Here, we assume all signals used the same scaler.
        # For more precise usage, you'd store a separate scaler per signal.
        denoised_signals[target_signal_index] = scaler_preprocessing_instance.inverse_transform(
            denoised_signals[target_signal_index].reshape(-1, 1)
        ).ravel()

    # Return only the requested target signal
    return denoised_signals[target_signal_index]