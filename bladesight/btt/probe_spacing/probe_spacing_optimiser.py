import numpy as np
from typing import Optional

def func_opt(
    theta: np.ndarray, 
    EO_array: np.ndarray,
    alpha_EO: Optional[np.ndarray] = None
) -> float:
    """
    Optimisation function, Sum of Squared Weighted Condition Numbers.
    From Equation (12) in "A Novel Method for the Design of Proximity Sensor Configuration for Rotor Blade Tip Timing" 
    by Diamond, Heyns 2018.

    Parameters
    ----------
    theta : np.ndarray
        Array of theta values. These are the absolute circumferential probe positions.
    EO_array : np.ndarray
        Array of EO values.
    alpha_EO : Optional[np.ndarray], optional
        Array of alpha values. These are the weighting factors for the EO values. Default is None.

    Returns
    -------
    float
        Cumulative sum of the square of the condition numbers for multiple EOs considered given a specific probe spacing.

    Examples
    --------
    >>> theta_test = np.array([1.67, 1.03, 0.15])
    >>> EO_array = np.array([1, 2, 3])
    >>> func_opt(theta_test, EO_array)
    123.456  # Example output
    """
    func_sum = 0

    if alpha_EO is None:
        alpha_EO = np.ones(len(EO_array))
    
    for i, EO in enumerate(EO_array):
        func_sum += (alpha_EO[i] * calculate_condition_number(probe_spacing = theta, EO = EO)) ** 2

    return func_sum
