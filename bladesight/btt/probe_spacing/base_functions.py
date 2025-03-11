import numpy as np

def get_EO(blade_frequency: float, omega: float) -> float:
    """
    Calculate the Engine Order (EO).

    Parameters
    ----------
    blade_frequency : float
        Blade frequency in radians per second.
    omega : float
        Shaft speed in radians per second.

    Returns
    -------
    float
        Engine Order (EO), which is the ratio of blade frequency to shaft speed.
    """
    return blade_frequency / omega


def get_PSR(EO: float, gamma: float) -> float:
    """
    Calculate the Probe Spacing Ratio (PSR).

    Parameters
    ----------
    EO : float
        Engine Order.
    gamma : float
        Probe spacing between the first and last probe in radians.

    Returns
    -------
    float
        Probe Spacing Ratio (PSR).
    """
    return EO * gamma / (2 * np.pi)


def get_arc_length(diameter: float, theta: float) -> float:
    """
    Calculate the arc length of a circle segment.

    Parameters
    ----------
    diameter : float
        Diameter of the circle.
    theta : float
        Central angle of the arc in radians.

    Returns
    -------
    float
        Arc length of the circle segment.
    """
    radius = diameter / 2
    arc_length = theta * radius
    return arc_length

def calculate_condition_number(probe_spacing : np.ndarray, EO : float):
    """
    Inputs:
    probe_spacing (np.ndarray): Array of circumferential probe positions in radians.
    EO (float): Engine order

    Returns:
    k (float): Condition number of the matrix Phi

    Note:
        - All condition numbers, for the EO's of interest, should be below 10 - but this is thumb sucky!
    """

    # print('len(probe_spacing) = ', len(probe_spacing))
    # print('probe_spacing.shape = ', probe_spacing.shape)
    Phi = np.ones((len(probe_spacing), 3))
    
    for i_probe, probe_location in enumerate(probe_spacing):
        # print('np.sin(EO * probe_location)', np.sin(EO * probe_location))
        Phi[i_probe, 0] = np.sin(EO * probe_location)
        Phi[i_probe, 1] = np.cos(EO * probe_location)

    k  = np.linalg.cond(Phi)
    return k