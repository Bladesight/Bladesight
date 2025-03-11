import numpy as np
from typing import Optional, List, Tuple
from scipy.optimize import NonlinearConstraint, LinearConstraint

from .base_functions import calculate_condition_number

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

def objective_function(
                    theta: np.ndarray, 
                    use_equidistant_spacing_bool: bool,
                    EO_array: np.ndarray, 
                    number_of_probes: Optional[int] = 4,
                    alpha_EO: Optional[np.ndarray] = None,
                    ) -> float:
    """
    Compute the objective function value for a given probe arrangement.

    The function applies custom weighting factors to certain Engine Orders (EO)
    and then computes the overall cost via a separate optimization routine.

    Parameters
    ----------
    theta : np.ndarray
        Shape (number_of_probes, ). The first index represents the absolute
        circumferential position of the first probe; subsequent indices define
        probe spacings.
    use_equidistant_spacing_bool : bool
        If True, imposes equidistant spacing for all probes. Otherwise, uses
        spacing defined by theta directly.
    EO_array : np.ndarray
        Array of Engine Orders (EOs).
    number_of_probes : int, optional
        The number of probes to consider. Defaults to 4 as this is the minimum number of probes recommended for tip timing.
    alpha_EO : Optional[np.ndarray], optional
        The weights for each EO. Defaults to None, which assigns a weight of 1 to all EOs.n.

    Returns
    -------
    float
        The calculated cost value based on weighted condition numbers 
        for the specified configuration.
    """    

    if alpha_EO is None:
        alpha_EO = np.ones(len(EO_array))

    if use_equidistant_spacing_bool == True:
        theta_array = [0]
        for i in range(number_of_probes-1):
            theta_array.append(theta[0])
        cost = func_opt(np.cumsum(theta_array), EO_array=EO_array, alpha_EO=alpha_EO) #Not sure about the cumsum here

    if use_equidistant_spacing_bool == False:
        cost = func_opt(np.cumsum(theta), EO_array=EO_array, alpha_EO=alpha_EO) #Not sure about the cumsum here

    return cost

def probe_position_constraints(
    x: np.ndarray,
    forbidden_intervals: List[Tuple[float, float]],
    min_spacing: float,
    max_spacing: float,
    # overall_max: float,
    delta: float = 1e-6
) -> np.ndarray:
    """
    Compute constraint values for probe spacings and their cumulative sensor positions.

    For each spacing xi in x, the following are enforced:
        - xi must be no smaller than min_spacing and no larger than max_spacing.
        - For cumulative positions p = cumsum(x) and for each forbidden interval (f_low, f_high),
            the constraint (p - f_low) * (p - f_high) - delta must be nonnegative, ensuring that
            p lies either below f_low or above f_high.

    Parameters
    ----------
    x : np.ndarray
        A one-dimensional numpy array of probe spacing values.
    forbidden_intervals : List[Tuple[float, float]]
        Forbidden intervals provided as (f_low, f_high) in radians.
    min_spacing : float
        The minimum allowable spacing (radians).
    max_spacing : float
        The maximum allowable spacing (radians).
    # overall_max : float
    #     The overall maximum sensor position (not directly used here).
    delta : float, optional
        A small margin to enforce strict inequality, by default 1e-6.

    Returns
    -------
    np.ndarray
        An array of constraint values that should be nonnegative when all conditions are met.
    """
    cons = []
    # Enforce individual spacing bounds.
    for xi in x:
        cons.append(xi - min_spacing)      # must be >= 0
        cons.append(max_spacing - xi)       # must be >= 0

    # Compute cumulative sensor positions.
    p = np.cumsum(x)
    # Enforce forbidden interval constraints.
    for pi in p:
        for (f_low, f_high) in forbidden_intervals:
            cons.append((pi - f_low) * (pi - f_high) - delta)
    return np.array(cons)

def create_probe_constraints(
    forbidden_intervals: List[Tuple[float, float]],
    min_spacing: float,
    max_spacing: float,
    overall_max: float,
    number_of_probes: int,
    delta: float = 1e-6
) -> List:
    """
    Create and return a list of constraints ensuring that the probe spacings
    satisfy individual bounds, avoid forbidden intervals, and do not exceed an overall maximum sensor position.

    The constraints are:
        1. A nonlinear constraint which enforces:
            - Each spacing xi satisfies: xi >= min_spacing and xi <= max_spacing.
            - The cumulative sensor positions (p = cumsum(x)) must lie outside every forbidden interval.
            For each forbidden interval (f_low, f_high) and for each sensor position p,
            the constraint (p - f_low) * (p - f_high) - delta >= 0 must hold.
        2. A linear constraint enforcing that the sum of spacings (i.e. the final sensor position)
            does not exceed overall_max:
            sum(x) <= overall_max.

    Parameters
    ----------
    forbidden_intervals : List[Tuple[float, float]]
        A list of forbidden intervals (in radians) provided as tuples (f_low, f_high).
    min_spacing : float
        Minimum spacing allowed between adjacent probes (in radians).
    max_spacing : float
        Maximum spacing allowed between adjacent probes (in radians).
    overall_max : float
        Maximum allowable cumulative sensor position (in radians).
    number_of_probes : int
        Total number of probe spacing variables.
    delta : float, optional
        A small margin used in the forbidden intervals constraint to enforce strict inequality, by default 1e-6.

    Returns
    -------
    List
        A list containing two constraints: a NonlinearConstraint and a LinearConstraint,
        which can be directly passed to an optimizer.
    """

    # Create the nonlinear constraint (for spacing bounds and forbidden intervals)
    nonlinear_constraint = NonlinearConstraint(
        fun=lambda x: probe_position_constraints(x,
                                                forbidden_intervals,
                                                min_spacing,
                                                max_spacing,
                                                # overall_max,
                                                delta
                                                ),
        lb=0,
        ub=np.inf
    )
    
    # Create the linear constraint enforcing sum(x) <= overall_max.
    linear_constraint = LinearConstraint(np.ones(number_of_probes), -np.inf, overall_max)
    
    return [nonlinear_constraint, linear_constraint]



### ------------- PSO implementation - Commented out to not add too many depndencies, but left in for interest ###
# def constraints_PSO(
#     theta: np.ndarray, 
#     number_of_probes: Optional[int] = 4
#     ) -> List[float]:
#     """
#     Calculate the constraints for the Particle Swarm Optimization (PSO) 
#     algorithm when determining probe spacing.

#     Parameters
#     ----------
#     theta : np.ndarray
#         Array of circumferential probe positions in radians. The array should 
#         have a size at least equal to ``number_of_probes``.
#     number_of_probes : int, optional
#         The total number of probes. Default is 4.

#     Returns
#     -------
#     list of float
#         A list of constraints derived from the differences between certain 
#         probe positions. In its current implementation, it collects the 
#         differences between theta[i] and theta[1] for i starting at 2.

#     Examples
#     --------
#     >>> import numpy as np
#     >>> theta_example = np.array([0.0, 0.2, 0.5, 0.7])
#     >>> constraints = constraints_PSO(theta_example, number_of_probes=4)
#     >>> print(constraints)
#     [0.3, 0.5]
#     """

#     constraints_list = []
#     for i in range(2, number_of_probes):
#         constraints_list.append(theta[i] - theta[1])# - theta[0])

#     # #PSR constraints --- this would be nice to include in the future, code is left here to show a possible starting point
#     # PSR = get_PSR(EO_array, (np.sum(theta) - theta[0]))
#     # constraints_list.append((1 - np.max(PSR))) #adding PSR constraint PSR < 100
#     # constraints_list.append(-0.30 + np.min(PSR)) #adding PSR constraint PSR > 30
#     # print("np.min(PSR) = {} np.max(PSR) = {}".format(np.min(PSR), np.max(PSR)))

#     # print("Constraints")
#     # print("np.cumsum(theta[0:])[-1]", np.cumsum(theta[0:])[-1])
#     # print("theta[0]", theta[0])
#     # print("np.cumsum(theta[0:])[-1] - theta[0])", (np.cumsum(theta[0:])[-1] - theta[0])) 
#     # print("-*"*100)

#     return constraints_list
### ------------- PSO implementation - Commented out to not add too many depndencies, but left in for interest ###