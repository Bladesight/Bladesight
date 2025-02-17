import numpy as np
import pandas as pd
from typing import List, Dict

from scipy.optimize import differential_evolution

def get_X(
        omega : np.ndarray,
        omega_n : float, 
        zeta: float,
        delta_st: float
    ) -> np.ndarray:
    """
    Compute the vibration amplitude of a single degree-of-freedom (SDoF) system.

    The amplitude is given by:
    X(ω) = δ_st / sqrt((1 - r^2)^2 + (2 ζ r)^2),
    where r = ω / ω_n.

    Parameters
    ----------
    omega : np.ndarray
        Excitation frequencies in rad/s.
    omega_n : float
        Natural frequency of the blade in rad/s.
    zeta : float
        Damping ratio of the blade vibration.
    delta_st : float
        Static deflection of the blade (often in µm).

    Returns
    -------
    np.ndarray
        Vibration amplitude at each frequency in omega.
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
    """
    Compute the phase of the blade vibration relative to the forcing function.

    The phase is given by:
    φ(ω) = arctan(2 ζ r / (1 - r^2)),
    where r = ω / ω_n.

    Parameters
    ----------
    omega : np.ndarray
        Excitation frequencies in rad/s.
    omega_n : float
        Natural frequency of the blade in rad/s.
    zeta : float
        Damping ratio of the blade vibration.

    Returns
    -------
    np.ndarray
        Phase of the blade vibration in radians at each frequency in omega.
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
    """
    Predict using the fitted SDoF samples at a proximity probe given the SDoF parameters.

    Parameters
    ----------
    omega_n : float
        Natural frequency of the SDoF system in rad/s.
    zeta : float
        Damping ratio of the SDoF system.
    delta_st : float
        Static deflection of the SDoF system.
    EO : int
        Engine order of vibration to fit.
    theta_sensor : float
        Angular position of the sensor on the rotor in radians.
    phi_0 : float
        Phase offset of the SDoF system in radians.
    arr_omega : np.ndarray
        Angular velocities of the rotor corresponding to each revolution.

    Returns
    -------
    np.ndarray
        Predicted SDoF samples.
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
    """
    Compute correction offsets for each sample based on the correction factors.

    Parameters
    ----------
    arr_omegas : np.ndarray
        The omega values for each sample.
    z_median : float
        The correction value at the median shaft speed.
    z_max : float
        The correction value at the max shaft speed.

    Returns
    -------
    np.ndarray
        The sample offsets for each SDoF sample.
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
    """
    Fit the SDoF parameters to multiple probes' data.

    Parameters
    ----------
    model_params : np.ndarray
        The SDoF fit method's model parameters. It includes a list of the following parameters:
            omega_n (float): The natural frequency of the SDoF system.
            ln_zeta (float): The damping ratio of the SDoF system.
            delta_st (float): The static deflection of the SDoF system.
            phi_0 (float): The phase offset of the SDoF system.
            And then the z_median and z_max for each probe.
            z_median (float): The amplitude offset at the median shaft speed.
            z_max (float): The maximum amplitude offset.
    tip_deflections_set : List[np.ndarray]
        The tip deflection data for each probe.
    arr_omega : np.ndarray
        The angular velocity of the rotor corresponding to the tip deflection data.
    EO : int
        Engine order of vibration to fit.
    theta_sensor_set : List[float]
        Angular positions of each probe relative to the start of the revolution.
    amplitude_scaling_factor : float, optional
        A scaling factor to weight the measured tip deflections. Use this value 
        to reward solutions that better capture the full amplitude of the tip deflections. 
        Defaults to 1.

    Returns
    -------
    np.ndarray
        The sum of squared error between the tip deflection data of each probe and the predicted tip deflections.
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
        Range of engine orders to consider, by default np.arange(1, 20).
    delta_st_max : int, optional
        Maximum static deflection for the optimizer bounds, by default 10.
    verbose : bool, optional
        If True, print progress updates, by default False.

    Returns
    -------
    Dict[str, Union[float, int]]
        Fitted parameters, including:
        - "omega_n": Natural frequency (in Hz).
        - "zeta": Damping ratio.
        - "delta_st": Static deflection.
        - "phi_0": Phase offset in radians.
        - "EO": Engine order with minimum error.
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