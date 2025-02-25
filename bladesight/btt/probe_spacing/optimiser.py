import numpy as np
from typing import Optional, List, Tuple
from .base_functions import calculate_condition_number, get_EO, get_PSR, get_arc_length

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

def constraints_PSO(
    theta: np.ndarray, 
    number_of_probes: Optional[int] = 4
    ) -> List[float]:
    """
    Calculate the constraints for the Particle Swarm Optimization (PSO) 
    algorithm when determining probe spacing.

    Parameters
    ----------
    theta : np.ndarray
        Array of circumferential probe positions in radians. The array should 
        have a size at least equal to ``number_of_probes``.
    number_of_probes : int, optional
        The total number of probes. Default is 4.

    Returns
    -------
    list of float
        A list of constraints derived from the differences between certain 
        probe positions. In its current implementation, it collects the 
        differences between theta[i] and theta[1] for i starting at 2.

    Examples
    --------
    >>> import numpy as np
    >>> theta_example = np.array([0.0, 0.2, 0.5, 0.7])
    >>> constraints = constraints_PSO(theta_example, number_of_probes=4)
    >>> print(constraints)
    [0.3, 0.5]
    """

    constraints_list = []
    for i in range(2, number_of_probes):
        constraints_list.append(theta[i] - theta[1])# - theta[0])

    # #PSR constraints --- this would be nice to include in the future, code is left here to show a possible starting point
    # PSR = get_PSR(EO_array, (np.sum(theta) - theta[0]))
    # constraints_list.append((1 - np.max(PSR))) #adding PSR constraint PSR < 100
    # constraints_list.append(-0.30 + np.min(PSR)) #adding PSR constraint PSR > 30
    # print("np.min(PSR) = {} np.max(PSR) = {}".format(np.min(PSR), np.max(PSR)))

    # print("Constraints")
    # print("np.cumsum(theta[0:])[-1]", np.cumsum(theta[0:])[-1])
    # print("theta[0]", theta[0])
    # print("np.cumsum(theta[0:])[-1] - theta[0])", (np.cumsum(theta[0:])[-1] - theta[0])) 
    # print("-*"*100)

    return constraints_list

def objective_function(
                    theta: np.ndarray, 
                    use_equidistant_spacing_bool: bool,
                    EO_array: np.ndarray, 
                    number_of_probes: Optional[int] = 4,
                    alpha_EO: Optional[np.ndarray] = None,
                    # d_min: Optional[float] = 21, 
                    # blade_outside_radius: Optional[float] = 440,
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
        The weights for each EO. Defaults to None, which assigns a weight of 1 to all EOs.
    # - d_min (Optional[int]): The minimum distance between probes.
    # - fan_outer_diameter (Optional[int]): The outer diameter of the fan.

    Returns
    -------
    float
        The calculated cost value based on weighted condition numbers 
        for the specified configuration.
    """
    # print('theta = ', theta) 
    # print('theta[-1] - theta[0]= ', np.cumsum(theta[0:])[-1] - theta[0]) 
    # print('theta.shape = ', theta.shape)    
    
    #EO weightings - prioritising EO8 and its harmonics
    # alpha_EO = np.ones(len(EO_array))
    # # alpha_EO[EO_array <= 32] = 20
    # alpha_EO[EO_array%4 == 0] = 20
    # alpha_EO[EO_array%8 == 0] = 30

    if alpha_EO is None:
        alpha_EO = np.ones(len(EO_array))

    #For equidistant spacing use the below code, otherwise commment it out
    if use_equidistant_spacing_bool == True:
        theta_array = [0]
        for i in range(number_of_probes-1):
            theta_array.append(theta[0])
        cost = func_opt(np.cumsum(theta_array), EO_array=EO_array, alpha_EO=alpha_EO) #Not sure about the cumsum here

    if use_equidistant_spacing_bool == False:
        cost = func_opt(np.cumsum(theta), EO_array=EO_array, alpha_EO=alpha_EO) #Not sure about the cumsum here

    return cost