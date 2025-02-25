"""
The probe spacing calculator module uses the method proposed in:
D. H. Diamond and P. S. Heyns, 'A Novel Method for the Design of Proximity Sensor Configuration for Rotor Blade Tip Timing',
Journal of Vibration and Acoustics, vol. 140, no. 6, p. 061003, Dec. 2018, doi: 10.1115/1.4039931.
"""
import numpy as np
import plotly.graph_objects as go
# import seaborn as sns
# import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
# import scipy.optimize as sciopt

from pyswarm import pso

import numpy as np
from typing import Optional, List, Tuple

from .probe_spacing_calculator import get_EO, get_PSR, get_arc_length
from .probe_spacing_optimiser import func_opt, constraints_PSO, objective_function

def perform_PSO(
    use_equidistant_spacing_bool: bool,
    EO_array: np.ndarray,
    number_of_probes: Optional[int] = 4,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    num_particles: Optional[int] = 5000,
    num_iterations: Optional[int] = 30,
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
    num_particles : int
        Number of particles in the swarm.
    num_iterations : int
        Maximum number of iterations to run the swarm optimization.

    Returns
    -------
    best_probe_spacing : np.ndarray
        The optimal probe spacing vector identified by the PSO algorithm.
    best_cost : float
        The cost associated with the optimal probe spacing.
    """
    best_probe_spacing, best_cost = pso(
        func=save_loss_progress_PSO,
        args=(use_equidistant_spacing_bool, EO_array, number_of_probes),
        lb=lower_bound,
        ub=upper_bound,
        swarmsize=num_particles,
        maxiter=num_iterations,
        f_ieqcons=constraints_PSO,
        minstep=1e-8,
        minfunc=1e-12,
        debug=True
    )
    return best_probe_spacing, best_cost