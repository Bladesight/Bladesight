import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

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
    EOs : List[int] = np.arange(1, 20),
    delta_st_max : int = 10,
    verbose : bool = False
) -> Dict[str, float]:
    """This function receives a blade tip deflection DataFrame, and returns 
    the SDoF fit model parameters after fitting.

    Args:
        df_blade (pd.DataFrame): The blade tip deflection DataFrame.
        n_start (int): The starting revolution number of the resonance 
            you want to fit.
        n_end (int): The ending revolution number of the resonance 
            you want to fit.
        EOs (List[int], optional): The list of EOs to search for. Defaults 
            to np.arange(1, 20).
        delta_st_max (int, optional): The maximum static deflection within our optimization 
            bounds. Defaults to 10.
        verbose (bool, optional): Whether to print the progress. Defaults to False.

    Returns:
        Dict[str, float]: The fitted model parameters.
    """
    df_resonance_window = df_blade.query(f"n >= {n_start} and n <= {n_end}")
    measured_tip_deflection_signals = [
        col 
        for col in df_resonance_window
        if col.endswith("_filt")
    ]
    PROBE_COUNT = len(measured_tip_deflection_signals)
    eo_solutions = []
    for EO in EOs:
        if verbose:
            print("NOW SOLVING FOR EO = ", EO, " of ", EOs)
        omega_n_min = df_resonance_window["Omega"].min() * EO
        omega_n_max = df_resonance_window["Omega"].max() * EO
        ln_zeta_min = np.log(0.0001)
        ln_zeta_max = np.log(0.3)
        delta_st_min = 0
        phi_0_min = 0
        phi_0_max = 2*np.pi
        bounds = [
            (omega_n_min, omega_n_max),
            (ln_zeta_min, ln_zeta_max),
            (delta_st_min, delta_st_max),
            (phi_0_min, phi_0_max),
        ]
        tip_deflections_set = []
        theta_sensor_set = []
        for i_probe in range(PROBE_COUNT):
            z_max = df_resonance_window[f"x_p{i_probe+1}_filt"].abs().max()
            z_min = -z_max
            bounds.extend(
                [
                    (z_min, z_max),
                    (z_min, z_max)
                ]
            )
            tip_deflections_set.append(
                df_resonance_window[f"x_p{i_probe+1}_filt"].values
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
                2
            ),
            seed=42
        )
        eo_solutions.append(multiple_probes_solution)
    best_EO_arg = np.argmin([solution.fun for solution in eo_solutions])
    best_EO = EOs[best_EO_arg]
    best_solution = eo_solutions[best_EO_arg]
    return {
        "omega_n" : best_solution.x[0] / (2*np.pi),
        "zeta" : np.exp(best_solution.x[1]),
        "delta_st" : best_solution.x[2],
        "phi_0" : best_solution.x[3],
        "EO" : best_EO,
    }