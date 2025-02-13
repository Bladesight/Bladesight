import numpy as np
from numba import njit
from typing import List, Optional, Tuple, Union, Callable


@njit
def _coeff_mat(x: np.ndarray, deg: int) -> np.ndarray:
    """
    Generate the coefficient matrix for polynomial fitting.

    Parameters
    ----------
    x : np.ndarray
        The input array of x values.
    deg : int
        The degree of the polynomial.

    Returns
    -------
    np.ndarray
        The coefficient matrix.
    
    References:
    -----------
    [1] Curve fitting code developed using: https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
    """
    mat_ = np.zeros(shape=(x.shape[0], deg + 1))
    const = np.ones_like(x)
    mat_[:, 0] = const
    mat_[:, 1] = x
    if deg > 1:
        for n in range(2, deg + 1):
            mat_[:, n] = x**n
    return mat_

@njit
def _fit_x_qr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve the least squares problem a·x = b using a QR decomposition.

    Parameters
    ----------
    a : np.ndarray
        The coefficient matrix.
    b : np.ndarray
        The dependent variable values.

    Returns
    -------
    np.ndarray
        The solution vector x that minimizes the residuals in a·x = b.
    """
    a_contig = np.ascontiguousarray(a)
    b_contig = np.ascontiguousarray(b)
    q, r = np.linalg.qr(a_contig)
    q_contig = np.ascontiguousarray(q)
    r_contig = np.ascontiguousarray(r)
    return np.linalg.solve(r_contig, np.dot(q_contig.T, b_contig)) # Use continguous arrays for Numba as its faster with np.dot

@njit
def _fit_x_weighted(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Solve a weighted least squares problem a·x = b using a diagonal weight matrix.

    Parameters
    ----------
    a : np.ndarray
        The coefficient matrix.
    b : np.ndarray
        The dependent variable values.
    w : np.ndarray
        Weights for each data point.

    Returns
    -------
    np.ndarray
        The solution vector x that minimizes the weighted residuals.
    """
    W = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        W[i, i] = w[i]
    a_conting = np.ascontiguousarray(a)
    b_conting = np.ascontiguousarray(b)
    a_w = np.dot(W, a_conting)#a)
    b_w = np.dot(W, b_conting)#b)
    a_w_contig = np.ascontiguousarray(a_w)
    b_w_contig = np.ascontiguousarray(b_w)
    q, r = np.linalg.qr(a_w_contig)
    q_contig = np.ascontiguousarray(q)
    r_contig = np.ascontiguousarray(r)

    return np.linalg.solve(r_contig, np.dot(q_contig.T, b_w_contig)) # Use continguous arrays for Numba as its faster with np.dot

@njit
def _fit_x_piecewise(a: np.ndarray, b: np.ndarray, segments: int) -> np.ndarray:
    """
    Fit multiple least squares segments and combine their solutions.

    Parameters
    ----------
    a : np.ndarray
        The coefficient matrix.
    b : np.ndarray
        The dependent variable values.
    segments : int
        Number of segments to split the data into.

    Returns
    -------
    np.ndarray
        Combined solution coefficients from each segment.
    """
    segment_size = len(a) // segments
    total_coeffs = (segments * (a.shape[1]))  # Total number of coefficients
    coeffs = np.zeros(total_coeffs)
    
    for i in range(segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i < segments - 1 else len(a)
        a_segment = a[start:end]
        b_segment = b[start:end]
        b_segment_conting = np.ascontiguousarray(b_segment)
        q, r = np.linalg.qr(a_segment)
        q_contig = np.ascontiguousarray(q)
        r_contig = np.ascontiguousarray(r)
        segment_coeffs = np.linalg.solve(r_contig, np.dot(q_contig.T, b_segment_conting))
        coeffs[i * a.shape[1]:(i + 1) * a.shape[1]] = segment_coeffs

    return coeffs

@njit
def _fit_x(
    a: np.ndarray,
    b: np.ndarray,
    method: str = 'qr',
    w: np.ndarray = None,
    segments: int = 1
    ) -> np.ndarray:
    """
    Solve the linear least squares problem ax = b using different methods.

    Parameters
    ----------
    a : np.ndarray
        The coefficient matrix.
    b : np.ndarray
        The dependent variable values.
    method : str
        The fitting method to use ('qr', 'weighted', 'piecewise').
    w : np.ndarray, optional
        The weights for each data point (used for 'weighted' method).
    segments : int, optional
        The number of segments to divide the data into (used for 'piecewise' method).

    Returns
    -------
    np.ndarray
        The solution to the least squares problem.
    """
    if method == 'qr':
        return _fit_x_qr(a, b)
    if method == 'weighted':
        if w is None:
            w = np.ones(len(b))  # Use equal weights if none are provided
        return _fit_x_weighted(a, b, w)
    if method == 'piecewise':
        return _fit_x_piecewise(a, b, segments)
    else:
        raise ValueError("Unsupported fitting method")

@njit
def fit_poly(
    x: np.ndarray,
    y: np.ndarray,
    deg: int,
    method: Optional[str] = 'qr',
    w: Optional[np.ndarray] = None,
    segments: Optional[int] = 1
    ) -> np.ndarray:
    """
    Fit a polynomial of given degree to the data.

    Parameters
    ----------
    x : np.ndarray
        The input array of x values.
    y : np.ndarray
        The input array of y values.
    deg : int
        The degree of the polynomial.
    method : str, optional
        The fitting method to use ('qr', 'weighted', 'piecewise'). Default is 'qr'.
    w : np.ndarray, optional
        The weights for each data point (used for 'weighted' method).
    segments : int, optional
        The number of segments to divide the data into (used for 'piecewise' method).

    Returns
    -------
    np.ndarray
        The coefficients of the fitted polynomial in descending order.

    References:
    -----------
    [1] Curve fitting code developed using: https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
    """

    a = _coeff_mat(x, deg)
    p = _fit_x(a, y, method = method, w = w, segments = segments)

    return p[::-1]

@njit
def eval_polynomial(P: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Compute polynomial P(x) where P is a vector of coefficients, highest
    order coefficient at P[0]. Uses Horner's Method.

    Parameters
    ----------
    P : np.ndarray
        The coefficients of the polynomial in descending order.
    x : np.ndarray
        The input array of x values.

    Returns
    -------
    np.ndarray
        The evaluated polynomial values.

    References:
    -----------
    https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
    """
    result = np.zeros_like(x)
    for coeff in P:
        result = x * result + coeff
    return result

@njit
def numba_linspace(start: float, stop: float, num: int) -> np.ndarray:
    """
    Generate linearly spaced values between `start` and `stop`.

    Parameters
    ----------
    start : float
        The starting value of the sequence.
    stop : float
        The end value of the sequence.
    num : int
        Number of values to generate.

    Returns
    -------
    np.ndarray
        Array of `num` linearly spaced values between `start` and `stop`.

    Examples
    --------
    >>> numba_linspace(0.0, 1.0, 5)
    array([0.  , 0.25, 0.5 , 0.75, 1.  ])
    """
    result = np.empty(num)
    step = (stop - start) / (num - 1)
    for i in range(num):
        result[i] = start + i * step
    return result

@njit
def eval_polynomial_derivative(coeffs: np.ndarray, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Evaluate the derivative of a polynomial at given x values.

    Parameters
    ----------
    coeffs : np.ndarray
        Polynomial coefficients in descending order.
    x : float or np.ndarray
        Points at which to evaluate the derivative.

    Returns
    -------
    float or np.ndarray
        Evaluated derivative values.
    """
    result = np.zeros_like(x)
    for i in range(1, len(coeffs)):
        c = coeffs[i]
        result += i * c * x**(i - 1)
    return result

@njit
def newton_raphson_minimize(
    y_test: float,
    poly_fitted_coeffs: np.ndarray,
    x0: Union[float, np.ndarray],
    tol: float = 1e-4,
    max_iter: int = 100
    ) -> float:
    """
    Use the Newton-Raphson method to find the x value that minimizes
    y_test - eval_polynomial(poly_fitted_coeffs, x).

    Parameters
    ----------
    y_test : float
        The target y value.
    poly_fitted_coeffs : np.ndarray
        Coefficients of the polynomial.
    x0 : float
        Initial guess for the x value.
    tol : float, optional
        Tolerance for convergence. Default is 1e-6.
    max_iter : int, optional
        Maximum number of iterations. Default is 100.

    Returns
    -------
    float
        The x value that minimizes the difference.
    """
    x = x0.astype(np.float64)  # Convert initial guess to float64
    for _ in range(max_iter):
        y = eval_polynomial(poly_fitted_coeffs, x)  # Evaluate polynomial at x
        dy_dx = eval_polynomial_derivative(poly_fitted_coeffs, x)  # Evaluate derivative at x
        if np.all(dy_dx == 0):  # Check if derivative is zero to avoid division by zero
            break
        x_new = x - (y - y_test) / dy_dx  # Newton-Raphson update
        if np.all(np.abs(x_new - x) < tol):  # Check for convergence
            break
        x = x_new  # Update x
    return x  # Return the refined x value

@njit
def threshold_crossing_hysteresis_curve_fit(
    arr_t: np.ndarray,
    arr_s: np.ndarray,
    threshold: float,
    hysteresis_height: float,
    poly_fit_bool: bool = False,
    poly_order: Optional[int] = 2,
    poly_fit_range: Optional[np.ndarray] = np.array([3, 3]),
    method: str = 'qr',
    w: np.ndarray = None,
    segments: int = 1,
    NR_tol: Optional[float] = 1e-4,
    NR_max_iter: Optional[int] = 500,
    n_est: Optional[float] = None,
    trigger_on_rising_edge: bool = True,
    ) -> np.ndarray:
    """A sequential threshold crossing algorithm that interpolates or performes a polynomial curve fit to find the Time of Arrival (ToA) between the two samples where the signal crosses the threshold.

    Args:
        arr_t (np.ndarray): The array containing the time values (time stamps of the signal).
        arr_s (np.ndarray): The array containing the signal voltage values
            corresponding to the time values. 
        threshold (float): The threshold value used to trigger the Time of Arrivals (ToAs), in the same
            units as the signal.
        hysteresis_height (float): The height of the hysteresis, in the same
            units as the signal.
        poly_fit_bool (bool, optional): Whether to use polynomial fitting for
            finding the ToA. Defaults to False. Note using a polynomial fit should be more accurate provided the polynomial fit is appropriate. Best to use if there are multiple samples near the threshold crossing.
        poly_order (int, optional): The order of the polynomial to fit. Only
            used if `poly_fit_bool` is True. Defaults to 2.
        poly_fit_range (np.ndarray[int], optional): The range of data points to use for
            polynomial fitting around the threshold crossing. Only used if
            "poly_fit_bool" is True. Defaults to 3 either size of the zero crossing time index.
        method : str
        The fitting method to use ('qr', 'weighted', 'piecewise').
        w : np.ndarray, optional
            The weights for each data point (used for 'weighted' method).
        segments : int, optional
            The number of segments to divide the data into (used for 'piecewise' method).
        NR_tol (float, optional): The tolerance for the Newton-Raphson solver. Defaults to 1e-4.
        NR_max_iter (int, optional): The maximum number of iterations for the Newton-Raphson solver. Defaults to 100.
        n_est (float, optional): The estimated number of ToAs in this
            signal. Defaults to None. This number is used to pre-allocate the array
            containing the ToAs. If this number is not provided, the array will
            be pre-allocated as the same dimension as arr_t and arr_s.
        trigger_on_rising_edge (bool, optional): Whether to trigger on the rising or falling edge of the pulse.
            Defaults to True.

    Returns:
        np.ndarray: An array containing the ToAs.
    References:
    -----------
    This function is adapted and relies on code from the Bladesight tutorial using the Bladesight Python package.
    [1] D. H. Diamond, “Introduction to Blade Tip Timing,” Bladesight Learn. Accessed: Feb. 12, 2024. [Online]. Available: docs.bladesight.com
    [2] Curve fitting code developed using: https://gist.github.com/kadereub/9eae9cff356bb62cdbd672931e8e5ec4
    """

    if trigger_on_rising_edge == True:
        threshold_lower = threshold - hysteresis_height
        trigger_state = True if arr_s[0] > threshold_lower else False

    if trigger_on_rising_edge == False:
        threshold_upper = threshold + hysteresis_height
        trigger_state = True if arr_s[0] < threshold_upper else False

    # Modifying the range to use for polynomial fitting around the threshold crossing
    poly_fit_range = np.array([poly_fit_range[0] - 1, poly_fit_range[1] - 1])

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

            if trigger_on_rising_edge == True and curr_sample <= threshold_lower:
                trigger_state = False

            if trigger_on_rising_edge == False and curr_sample >= threshold_upper:
                trigger_state = False

        if trigger_state is False:
            if trigger_on_rising_edge == True and curr_sample >= threshold:
                trigger_state = True
                # Interpolate the ToA
                if poly_fit_bool == False:
                    arr_toa[i_toa] = arr_t[i_sample - 1] + (
                        arr_t[i_sample] - arr_t[i_sample - 1]
                    ) * (threshold - prev_sample) / (curr_sample - prev_sample)
                    i_toa += 1

                if poly_fit_bool == True:
                    # Interpolate and Curve Fit the ToA
                    # poly_fitted_coeffs = fit_poly(arr_t[i_sample - poly_fit_range[0] : i_sample + poly_fit_range[1]], arr_s[i_sample - poly_fit_range[0] : i_sample + poly_fit_range[1]], deg = int(poly_order))
                    arr_t_fit = arr_t[i_sample - poly_fit_range[0] : i_sample + poly_fit_range[1]]
                    arr_s_fit = arr_s[i_sample - poly_fit_range[0] : i_sample + poly_fit_range[1]]

                    # print("arr_t_fit: ", arr_t_fit, "\n arr_s_fit: ", arr_s_fit)
                    poly_fitted_coeffs = fit_poly(arr_t_fit, arr_s_fit, deg = int(poly_order), method = method, w = w, segments = segments)
                    # print("poly_fitted_coeffs: ", poly_fitted_coeffs)

                    # Using a Newton Raphson solver to find the x value that minimizes the difference between the threshold and the polynomial fit
                    interpolated_ToA_guess = arr_t[i_sample - 1] + (
                        arr_t[i_sample] - arr_t[i_sample - 1]
                    ) * (threshold - prev_sample) / (curr_sample - prev_sample) # Use the interpolated point as an initial guess since it does not cost a lot computationally
                    
                    arr_toa[i_toa] = newton_raphson_minimize(np.array([threshold]), poly_fitted_coeffs, np.array([interpolated_ToA_guess]), tol = NR_tol, max_iter = NR_max_iter)[0] # Note that all the inputs need to be the same variable type (in this case numpy arrays) for Numba to work properly.

                    i_toa += 1
            
            if trigger_on_rising_edge == False and curr_sample <= threshold:
                trigger_state = True
                # Interpolate the ToA
                if poly_fit_bool == False:
                    arr_toa[i_toa] = arr_t[i_sample - 1] + (
                        arr_t[i_sample] - arr_t[i_sample - 1]
                    ) * (threshold - prev_sample) / (curr_sample - prev_sample)
                    i_toa += 1

                if poly_fit_bool == True:
                    # Interpolate and Curve Fit the ToA
                    # poly_fitted_coeffs = fit_poly(arr_t[i_sample - poly_fit_range[0] : i_sample + poly_fit_range[1]], arr_s[i_sample - poly_fit_range[0] : i_sample + poly_fit_range[1]], deg = int(poly_order))
                    arr_t_fit = arr_t[i_sample - poly_fit_range[0] : i_sample + poly_fit_range[1]]
                    arr_s_fit = arr_s[i_sample - poly_fit_range[0] : i_sample + poly_fit_range[1]]

                    # x_fit =  np.ones_like(arr_t_fit) *arr_t_fit
                    poly_fitted_coeffs = fit_poly(arr_t_fit, arr_s_fit, deg = int(poly_order), method = method, w = w, segments = segments)

                    # Using a Newton Raphson solver to find the x value that minimizes the difference between the threshold and the polynomial fit
                    interpolated_ToA_guess = arr_t[i_sample - 1] + (
                        arr_t[i_sample] - arr_t[i_sample - 1]
                    ) * (threshold - prev_sample) / (curr_sample - prev_sample) # Use the interpolated point as an initial guess since it does not cost a lot computationally

                    arr_toa[i_toa] = newton_raphson_minimize(np.array([threshold]), poly_fitted_coeffs, np.array([interpolated_ToA_guess]), tol = NR_tol, max_iter = NR_max_iter)[0] # Note that all the inputs need to be the same variable type (in this case numpy arrays) for Numba to work properly.

                    i_toa += 1

        # Update the previous sample value
        prev_sample = curr_sample

    # Return the array containing the ToAs
    return arr_toa[:i_toa]