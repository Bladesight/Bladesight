import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from scipy.optimize import differential_evolution

def get_X(
        omega : np.ndarray,
        omega_n : float, 
        zeta: float,
        delta_st: float
    ) -> np.ndarray:
    """
    This function returns the vibration amplitude of 
    the blade vibration.
    
    x(ω) = 	δ_st / sqrt( (1 - r**2)**2 + (2*ζ*r)**2)

    where:

    r = ω/ω_0

    Args:
        omega (np.ndarray): The excitation frequencies in rad/s.
        omega_n (float): The natural frequency of the blade in rad/s.
        zeta (float): The damping ratio of the blade vibration.
        delta_st (float, optional): The static deflection of the blade. 
            This value is usually given in units of µm.

    Returns:
        np.ndarray: The amplitude of the blade vibration in the
            same units as delta_st.
    """
    r = omega / omega_n
    return (
        delta_st 
        / np.sqrt(
            (1 - r**2)**2 
            + (2*zeta*r)**2
        )
    )

def get_phi(
    omega : np.ndarray, 
    omega_n : float, 
    zeta: float
) -> np.ndarray:
    """Get the phase between the tip deflection and 
        the forcing function. 

    φ(ω) = arctan(2*ζ*r /  (1 - r**2))
    
    where:
    r = ω/ω_n

    Args:
        omega (np.ndarray): The excitation frequencies in rad/s.
        omega_0 (float): The natural frequency of the blade in rad/s.
        delta (float): The damping ratio of the blade vibration.

    Returns:
        np.ndarray: The phase of the blade vibration in rad.
    """
    r = omega / omega_n
    return np.arctan2(2 * zeta * r,1 - r**2)

def predict_sdof_samples(
    omega_n : float,
    zeta : float,
    delta_st : float,
    EO : int,
    theta_sensor : float,
    phi_0 : float,
    arr_omega : np.ndarray
) -> np.ndarray:
    """ This function determined the predicted SDoF fit
    samples at a proximity probe given the SDoF parameters.

    Args:
        omega_n (float): The natural frequency of the SDoF system.
        zeta (float): The damping ratio of the SDoF system.
        delta_st (float): The static deflection of the SDoF system.
        phi_0 (float): The phase offset of the SDoF system.
        EO (int): The EO of vibration you want to fit.
        theta_sensor (float): The sensor's angular position on the rotor.
        phi_0 (float): The phase offset of the SDoF system.
        arr_omega (np.ndarray): The angular velocity of the rotor corresponding
            to each revolution for which we want to predict the SDoF samples.

    Returns:
        np.ndarray: The predicted SDoF samples.
    """
    X = get_X(arr_omega*EO, omega_n, zeta, delta_st)  
    phi = get_phi(arr_omega*EO, omega_n, zeta)
    predicted_tip_deflections = X * np.cos(theta_sensor * EO - phi + phi_0)
    return predicted_tip_deflections

def get_correction_values(
    arr_omegas : float,
    z_median : float,
    z_max : float, 
) -> np.ndarray:
    """This function calculates the correction values for each sample
    based on the correction factors.

    Args:
        arr_omegas (float): The omega values for each sample.
        z_median (float): The correction value at the median shaft speed.
        z_max (float): The correction value at the max shaft speed.

    Returns:
        np.ndarray: The sample offsets for each sample.
    """
    omega_median = np.median(arr_omegas)
    omega_max = np.min(arr_omegas)
    m = (
        z_max
        - z_median
    ) / (
        omega_max 
        - omega_median
    )
    b = z_median - m * omega_median
    correction_values = m * arr_omegas  + b
    return correction_values

def SDoF_loss_multiple_probes(
        model_params : np.ndarray,
        tip_deflections_set : List[np.ndarray],
        arr_omega : np.ndarray, 
        EO : int, 
        theta_sensor_set : List[float],
        amplitude_scaling_factor : float = 1
    ) -> np.ndarray:
    """ This function fits the SDoF parameters to 
        multiple probes' data.

    Args:
        model_params (np.ndarray): The SDoF fit method's model parameters. It
            includes a list of the following parameters:
            
            omega_n (float): The natural frequency of the SDoF system.
            ln_zeta (float): The damping ratio of the SDoF system.
            delta_st (float): The static deflection of the SDoF system.
            phi_0 (float): The phase offset of the SDoF system.
            And then the z_median and z_max for each probe.
                z_median (float): The amplitude offset at the 
                    median shaft speed.
                z_max (float): The maximum amplitude offset.

        tip_deflections_set (List[np.ndarray]): The tip deflection data for each probe.
        arr_omega (np.ndarray): The angular velocity of the rotor corresponding
            to the tip deflection data.
        EO (int): The EO of vibration you want to fit.
        theta_sensor_set (List[float]): Each sensor's angular position 
            relative to the start of the revolution.
        amplitude_scaling_factor (float, optional): A scaling factor to
            weight the measured tip deflections. Defaults to 1. Use this value
            to reward solutions that better capture the full amplitude of the
            tip deflections.

    Returns:
        np.ndarray: The sum of squared error between 
            the tip deflection data of each probe and
            the predicted tip deflections.
    """
    omega_n, ln_zeta, delta_st, phi_0, *correction_factors = model_params
    zeta = np.exp(ln_zeta)
    error = 0
    for i_probe, arr_tip_deflections in enumerate(tip_deflections_set):    
        theta_sensor = theta_sensor_set[i_probe]
        predicted_tip_deflections = predict_sdof_samples(
            omega_n, zeta, delta_st, EO, theta_sensor, phi_0, arr_omega
        )
        z_median = correction_factors[i_probe*2]
        z_max = correction_factors[i_probe*2+1]
        arr_tip_deflection_corrections = get_correction_values(
            arr_omega, z_median, z_max
        )
        arr_tip_deflections_corrected = (
            arr_tip_deflections
            + arr_tip_deflection_corrections
        )
        error += np.sum(
            np.abs(arr_tip_deflections_corrected)**amplitude_scaling_factor
            *(
                arr_tip_deflections_corrected
                - predicted_tip_deflections
            )**2
        )
    return error

def perform_SDoF_fit(
    df_blade : pd.DataFrame,
    n_start : int,
    n_end : int,
    EOs : Optional[List[int]] = np.arange(1, 20),
    omega_n_bounds : Optional[List[float]] = [None, None],
    ln_zeta_bounds : Optional[List[float]] = [np.log(0.0001), np.log(0.3)],
    delta_st_bounds : Optional[List[float]] = [0, 10],
    phi_0_bounds : Optional[List[float]] = [0, 2*np.pi],
    signal_suffix : Optional[str] = "_filt" ,
    amplitude_scaling_factor : float = 1,
    differential_evolution_optimiser_kwargs: Optional[dict] = {'seed': 42},
    verbose : bool = False
) -> Dict[str, float]:
    """
    Fit an SDoF model to blade tip deflection data over a specified revolution range.

    Parameters
    ----------
    df_blade : pd.DataFrame
        The blade tip deflection DataFrame.
    n_start : int
        The starting revolution number of the resonance to be used in the fit.
    n_end : int
        The ending revolution number of the resonance to be used in the fit.
    EOs : List[int], optional
        Range of engine orders to consider, defaults to np.arange(1, 20).
    omega_n_bounds : List[float], optional
        [min, max] bounds for resonant speed Ωₙ (radians/second). If None, set per EO as
        (min(df_blade['Omega'])*EO, max(df_blade['Omega'])*EO).
    zeta_bounds : list of float, optional
        [min, max] bounds for damping ratio ζ, default to: [0.0001, 0.3].
    delta_st_bounds : list of float, optional
        [min, max] bounds for static deflection δₛₜ, , defaults to [0, 10].
    phi_0_bounds : list of float, optional
        [min, max] bounds for phase offset φ₀ in radians, defaults to [0, 2π].
    signal_suffix : str, optional
        Suffix appended to deflection column names, defaults to "_filt".
    amplitude_scaling_factor : float, optional
        Scaling factor for amplitude weighting, defaults to 1.
    differential_evolution_optimiser_kwargs : dict, optional
        Additional arguments for the differential evolution optimizer, defaults to {'seed': 42}.
        If you want to use the default settings, pass an empty dictionary.
    verbose : bool, optional
        If True, print progress updates, defaults to False.
    Returns:
        Dict[str, float]: The fitted model parameters.
    Returns
    -------
    Dict[str, Union[float, int]]
        Fitted parameters, including:
        - "omega_n": Natural frequency (in Hz).
        - "zeta": Damping ratio.
        - "delta_st": Static deflection.
        - "phi_0": Phase offset in radians.
        - "EO_best": Engine order with minimum error.
        - "EO_errors": Dictionary of errors for each engine order.
    """
    df_resonance_window = df_blade.query(f"n >= {n_start} and n <= {n_end}")
    measured_tip_deflection_signals = [
        col 
        for col in df_resonance_window
        # if col.endswith("_filt")
        if col.startswith("x_p") and col.endswith(signal_suffix)
    ]
    PROBE_COUNT = len(measured_tip_deflection_signals)
    EO_solutions = []
    EO_errors = {}  # collect each EO's loss
    for EO in EOs:
        if verbose:
            print("NOW SOLVING FOR EO = ", EO, " of ", EOs)
        
        if omega_n_bounds.__contains__(None):
            omega_n_bounds[0] = df_resonance_window["Omega"].min() * EO
            omega_n_bounds[1] = df_resonance_window["Omega"].max() * EO
        # if omega_n_bounds.__contains__(None) == False:
        #     omega_n_bounds[0] = omega_n_bounds[0]
        #     omega_n_bounds[1] = omega_n_bounds[1]

        bounds = [
            (omega_n_bounds[0], omega_n_bounds[1]),
            (ln_zeta_bounds[0], ln_zeta_bounds[1]),
            (delta_st_bounds[0], delta_st_bounds[1]),
            (phi_0_bounds[0], phi_0_bounds[1]),
        ]
        tip_deflections_set = []
        theta_sensor_set = []
        for i_probe in range(PROBE_COUNT):
            z_max = df_resonance_window[f"x_p{i_probe+1}"+signal_suffix].abs().max()
            z_min = -z_max
            bounds.extend(
                [
                    (z_min, z_max),
                    (z_min, z_max)
                ]
            )
            tip_deflections_set.append(
                df_resonance_window[f"x_p{i_probe+1}"+signal_suffix].values
            )
            theta_sensor_set.append(
                df_resonance_window[f"AoA_p{i_probe+1}"].median()
            )
        multiple_probes_solution = differential_evolution(
            func = SDoF_loss_multiple_probes,
            bounds=bounds,
            args=(
                tip_deflections_set,
                df_resonance_window['Omega'].values,
                EO,
                theta_sensor_set,
                amplitude_scaling_factor,
            ),
            **differential_evolution_optimiser_kwargs
        )
        EO_solutions.append(multiple_probes_solution)
        EO_errors[f"EO{EO}"] = multiple_probes_solution.fun
    
    # Select the best EO
    best_EO_arg = np.argmin([solution.fun for solution in EO_solutions])
    best_EO = EOs[best_EO_arg]
    best_solution = EO_solutions[best_EO_arg]
    return {
        "omega_n" : best_solution.x[0] / (2*np.pi),
        "zeta" : np.exp(best_solution.x[1]),
        "delta_st" : best_solution.x[2],
        "phi_0" : best_solution.x[3],
        "EO_best" : best_EO,
        "EO_errors" : EO_errors,
    }