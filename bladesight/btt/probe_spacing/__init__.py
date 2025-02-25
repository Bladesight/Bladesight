"""
The probe spacing calculator module uses the method proposed in:
D. H. Diamond and P. S. Heyns, 'A Novel Method for the Design of Proximity Sensor Configuration for Rotor Blade Tip Timing',
Journal of Vibration and Acoustics, vol. 140, no. 6, p. 061003, Dec. 2018, doi: 10.1115/1.4039931.
"""
import numpy as np
import plotly.graph_objects as go
from typing import Optional, List, Tuple
from pyswarm import pso
from typing import Optional, List, Tuple
import plotly.graph_objects as go

from .probe_spacing_calculator import get_EO, get_PSR, get_arc_length
from .probe_spacing_optimiser import func_opt, constraints_PSO, objective_function#, save_loss_progress_PSO

loss_list_PSO = [] # Need this global variable to store the loss values during PSO iterations
def save_loss_progress_PSO(
    theta: np.ndarray,
    use_equidistant_spacing_bool,
    EO_array: [np.ndarray],
    number_of_probes: Optional[int] = 4
    #,d_min: Optional[float] = 21, blade_outside_radius: Optional[float] = 440
    ): 
    """
    Record the progress of the PSO algorithm by computing and storing the 
    current cost value in the global loss_list_PSO.

    Parameters
    ----------
    theta : numpy.ndarray
        Array of angles (in radians). The first element typically represents
        the absolute circumferential position of the first probe; subsequent
        elements indicate the probe spacings.
    use_equidistant_spacing_bool : bool
        If True, probes are spaced equidistantly. If False, uses the positions
        provided in ``theta``.
    EO_array : numpy.ndarray
        Array of engine orders (EOs).
    number_of_probes : int, optional
        The number of probes. Default is 4.

    Returns
    -------
    float
        The computed cost value based on the provided parameters.

    Notes
    -----
    This function updates the global variable ``loss_list_PSO`` by appending 
    the newly computed cost for each call during a PSO iteration.

    Examples
    --------
    >>> import numpy as np
    >>> # Suppose we have 3 probes, spaced at [0, 0.5, 0.8] radians
    >>> theta_test = np.array([0.0, 0.5, 0.8])
    >>> # Engine orders of interest
    >>> EO_array = np.array([1, 2, 4, 8])
    >>> # Call the function to record the cost in global loss_list_PSO
    >>> cost_value = save_loss_progress_PSO(
    ...     theta_test,
    ...     use_equidistant_spacing_bool=False,
    ...     EO_array=EO_array,
    ...     number_of_probes=3
    ... )
    >>> print(cost_value)
    12.3456  # Example output
    """
    cost = objective_function(theta = theta, use_equidistant_spacing_bool = use_equidistant_spacing_bool, EO_array = EO_array, number_of_probes = number_of_probes)#, d_min = 21, blade_outside_radius = 440)
    global loss_list_PSO
    loss_list_PSO.append(cost)
    return cost


def perform_PSO(
    use_equidistant_spacing_bool: bool,
    EO_array: np.ndarray,
    number_of_probes: Optional[int] = 4,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    number_of_particles: Optional[int] = 5000,
    number_of_iterations: Optional[int] = 30,
    minstep: Optional[float] = 1e-8,
    minfunc: Optional[float] = 1e-12,
    debug: Optional[bool] = False,
) -> Tuple[np.ndarray, float]:
    """
    Perform Particle Swarm Optimization (PSO) to find the optimal probe spacing.

    Parameters
    ----------
    use_equidistant_spacing_bool : bool
        If True, probes are spaced equidistantly. If False, uses specified spacings in theta.
    EO_array : np.ndarray
        Array of engine orders (EOs).
    number_of_probes : int, optional
        Number of probes to consider. If None, defaults to the underlying function's default.
    lower_bound : np.ndarray
        Lower bound for each dimension of the swarm search space.
    upper_bound : np.ndarray
        Upper bound for each dimension of the swarm search space.
    number_of_particles : int
        Number of particles in the swarm.
    number_of_iterations : int
        Maximum number of iterations to run the swarm optimization.
    minstep : float
        Minimum step size for the swarm optimization.
    minfunc : float
        Minimum function value for the swarm optimization.
    debug : bool
        If True, print debug information.
    Returns
    -------
    best_probe_spacing : np.ndarray
        The optimal probe spacing vector identified by the PSO algorithm.
    best_cost : float
        The cost associated with the optimal probe spacing.
    loss_list_PSO : list
        A list of cost values during the PSO iterations.
    """
    best_probe_spacing, best_cost = pso(
        func=save_loss_progress_PSO,
        args=(use_equidistant_spacing_bool, EO_array, number_of_probes),
        lb=lower_bound,
        ub=upper_bound,
        swarmsize=number_of_particles,
        maxiter=number_of_iterations,
        # f_ieqcons=constraints_PSO, #Rather control fixtures through lower_bound and upper_bound variables
        minstep=minstep,
        minfunc=minfunc,
        debug=debug
    )
    return best_probe_spacing, best_cost, loss_list_PSO


def plot_best_PSO_loss_per_iter(
    loss_list_PSO: List[float],
    number_of_particles: int
) -> None:
    """
    Plot the best (minimum) Particle Swarm Optimization (PSO) loss value 
    found at each iteration.

    The function examines subsets of the global loss array, corresponding to 
    each swarm iteration (i.e., blocks of size ``number_of_particles``). 
    It extracts the minimum cost value per iteration and visualizes how
    that minimum evolves over the number of iterations.

    Parameters
    ----------
    loss_list_PSO : list of float
        A global list that stores the loss values computed during PSO. 
        The list is expected to have length ``(num_iterations + 1) * number_of_particles`` 
        or similar, with a block of size ``number_of_particles`` for each iteration.
    number_of_particles : int
        Number of particles in the swarm.

    Returns
    -------
    None
        This function does not return anything. It displays a plot 
        of the best (minimum) loss value at each iteration.

    Examples
    --------
    >>> loss_list_example = [5.0, 4.0, 6.0, 4.5, 3.9, 3.5, 3.2, 2.9, 3.0]
    >>> number_of_particles_example = 3
    >>> plot_best_PSO_loss_per_iter(loss_list_example, number_of_particles_example)
    # A plot is displayed showing the evolution of the minimum loss value 
    # over the PSO iterations.
    """
    min_losses_PSO = []

    print("len(loss_list_PSO):", len(loss_list_PSO))
    num_iterations_completed = int(len(loss_list_PSO[number_of_particles:]) / number_of_particles)

    # Minimum loss for the 0th iteration block
    min_losses_PSO.append(
        np.min(loss_list_PSO[0 * number_of_particles:(0 + 1) * number_of_particles])
    )

    # Track minimum losses for subsequent iterations
    for i in range(1, num_iterations_completed + 1):
        current_min = np.min(loss_list_PSO[i * number_of_particles:(i + 1) * number_of_particles])
        if current_min < min_losses_PSO[-1]:
            min_losses_PSO.append(current_min)
        else:
            min_losses_PSO.append(min_losses_PSO[-1])

    # Plot using Plotly
    iterations = np.arange(len(min_losses_PSO))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=min_losses_PSO,
            mode='lines+markers',
            name='Min Loss per Iteration'
        )
    )
    fig.update_layout(
        title='Best PSO Loss per Iteration',
        xaxis_title='Iteration',
        yaxis_title='Minimum Loss Value',
        template='plotly_white'
    )
    fig.show()