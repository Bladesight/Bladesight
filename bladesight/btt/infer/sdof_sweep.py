"""
This module contains the functions proposed by:
[1] F. Zhi et al., `Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate`,
Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.

The notation in this module follows the one used in the reference [1].

Please also see the bladesight dataset linked below for varying ramp rates that was conducted before finding Reference [1]:

https://docs.bladesight.com/tutorials/datasets/diamond_et_al_2024_vary_ramp_rates_45/

"""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from typing import List, Dict


from .sdof import get_phi

# ------------------ Frequency Sweep Parameter Method ------------------ #
def get_eta(EO: int, Q: float, a: float, 
            #f_n: float,
            Omega_n:float) -> float:
    """
    Compute the non-dimensional frequency sweep parameter η.

    Equation (11) in Reference [1]:
        η = EO * ((Q^2 * a) / (60 * f_n^2))

    Parameters
    ----------
    EO : int
        Engine order.
    Q : float
        Quality factor (dimensionless).
    a : float
        Frequency sweep rate (acceleration of rotating speed).
    # f_n : float
    #     Natural frequency (in Hz or consistent units).
    Omega_n : float
        The rotating speed at resonance (in rad/s).

    Returns
    -------
    float
        The computed non-dimensional Frequency sweep parameter η.

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.
    """
    # return EO*((Q**2) * a)/(60*(f_n**2))
    return ((Q**2) * a)/(60*EO*(Omega_n**2)) # Equation 11 in Reference [1], NOT using f_n as its just an extra parameter to pass in unnecessarily

def get_A(eta: float) -> float:
    """
    Compute the fraction of the peak amplitude factor, A, based on the non-dimensional parameter η.

    Equation (13) in Reference [1]:
        A(η) = 1 - exp(-2.86 * η^(-0.455))

    Parameters
    ----------
    eta : float
        The non-dimensional frequency parameter η.

    Returns
    -------
    float
        The peak amplitude factor A.

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.
    
    Notes
    -----
    Note reference [1] says that Equation (13) improves on Equation (12), therefore only Equation (13) is implemented here.
    """
    return 1 - np.exp(-2.86*eta**(-0.455)) # Equation 13 in Reference [1]

def get_f(eta: float) -> float:
    """
    Compute the normalised frequency error, f, based on the non-dimensional parameter η.

    Equation (13) in Reference [1]:
        f(η) = 0.518 * η^(0.576)

    Parameters
    ----------
    eta : float
        The non-dimensional frequency parameter η.

    Returns
    -------
    float
        The normalised frequency error f.

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.

    Notes
    -----
    Note reference [1] says that Equation (13) improves on Equation (12), therefore only Equation (13) is implemented here.
    """    
    return 0.518*eta**0.576 # Equation 13 in Reference [1]

def get_zeta(eta: float) -> float:
    """
    Compute the fraction of damping ratio, ζ, based on the non-dimensional parameter η.

    Equation (14) in Reference [1]:
        ζ(η) = 1 / A

    where A is computed by `get_A(eta)`.

    Parameters
    ----------
    eta : float
        The non-dimensional frequency parameter η.

    Returns
    -------
    float
        The fraction of damping ratio ζ.

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.
    """
    return 1/get_A(eta) # Equation 14 in Reference [1]

# def get_Q(eta: float) -> float:
#     """
#     Compute the quality factor, Q, based on the non-dimensional parameter η.

#     Section 3.2 in Reference [1]:
#         Q = 1/(2*ζ)

#     Parameters
#     ----------
#     eta : float
#         The non-dimensional frequency parameter η.

#     Returns
#     -------
#     float
#         The quality factor Q.

#     References
#     ----------
#     [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
#         Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.
#     """
#     return 1/(2*get_zeta(eta))

def get_frequency_sweep_rate_a(speed_at_start: float, speed_at_end: float, time_at_start: float, time_at_end: float) -> float:
    """
    Compute the frequency sweep rate, a, based on the start and end speeds and times.

    Parameters:
    -----------
    speed_at_start : float
        The rotating speed at the start of the frequency sweep (in rad/s).
    speed_at_end : float
        The rotating speed at the end of the frequency sweep (in rad/s).
    time_at_start : float
        The time at the start of the frequency sweep (in seconds).
    time_at_end : float
        The time at the end of the frequency sweep (in seconds).

    Returns:
    --------
    float
        The frequency sweep rate, a (in rad/s^2).

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.

    Notes:
    ------
    In Section 3.2 of Reference [1], the frequency sweep rate should be smaller than 216*(f_n**2)*ζ**2.    
    """
    return (speed_at_end - speed_at_start) / (time_at_end - time_at_start)

def predict_sdof_samples_sweep(
    EO : int,
    theta_sensor : float,
    phi_0 : float,
    # Reference [1] parameters:
    Q: float,
    # a: float,
    # f_n: float,
    A_max: float,
    C: float,
    Omega_n: float, # Moved Omega_n up here so it can be optimised


    Omega_arr: np.ndarray,
    time_arr: np.ndarray,
    
    verbose: bool = False
) -> np.ndarray:
    """
    Predict blade tip deflections using a frequency sweep parameter method based on a
    Single Degree-of-Freedom (SDoF) model.

    This function implements a modified SDoF model that incorporates the frequency sweep 
    effects described in reference [1]. In this formulation, several intermediate parameters 
    are computed:
    
    - The non-dimensional parameter η is computed using `get_eta`.
    - The amplitude factor A, frequency factor f, and revised damping ratio ζ are then derived 
    from η using `get_A`, `get_f`, and `get_zeta`, respectively.
    - The predicted tip deflection is then computed via a formulation involving:
    
        numerator_term1 = A * A_max
        numerator_term2 = v * cos(EO * theta_sensor + phi_0) + sin(EO * theta_sensor + phi_0)

        v = Q/ζ * [ (1 - ((Omega/(f/Q + 1)) * Omega_n)**2) / (((Omega/(f/Q + 1)) * Omega_n)**2) ], defined below Equation (16) in Reference [1]

        denominator_term1 = Omega / ((f/Q + 1) * Omega_n)
        denominator_term2 = (v**2 + 1)

        predicted_tip_deflections = (numerator_term1 / denominator_term1) * (numerator_term2 / denominator_term2) + C

    Parameters
    ----------
    EO : int
        Engine order.
    theta_sensor : float
        The sensor angle in radians (absolute circumferential position of the probe).
    phi_0 : float
        The initial phase of the excitation force in radians.

    Reference [1] parameters:
    Q : float
        Quality factor (dimensionless).
    a : float
        Frequency sweep rate (acceleration of rotating speed).
    # f_n : float
    #     Natural frequency (in Hz or consistent units).
    A_max : float
        Resonant maximum amplitude of the blade vibration.
    C : float
        Constant vibration offset term used when calculating the predicted tip deflections.
    Omega_n : float
        The rotating speed at resonance (in rad/s).
    Omega_arr : np.ndarray
        Array of excitation frequencies (in rad/s).
    time_arr : np.ndarray
        Array of corresponding times (s).
    verbose : bool, optional
        If True, prints intermediate sweep rate. Defaults to False.


    Returns
    -------
    np.ndarray
        The predicted tip deflection values (in the same units as A_max) for each 
        excitation frequency in `arr_omega`.

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi:10.1016/j.measurement.2022.111681.

    Notes
    -----
    - This formulation is based on the frequency sweep parameter method incorporating frequency-dependent 
    loss mechanisms.
    """
    a = get_frequency_sweep_rate_a(
                                speed_at_start = Omega_arr[0], speed_at_end = Omega_arr[-1], 
                                time_at_start = time_arr[0], time_at_end = time_arr[-1]
                                )
    
    if a == 0:
        raise ValueError("Frequency sweep rate is zero. Please check the input data for valid speed and time values.")

    if a < 0:
        a = abs(a) # Ensure a is positive for the calculations
        print("Warning: Frequency sweep rate is negative. Taking the absolute value of the frequency sweep rate to avoid potential issues with calculating the frequency sweep parameter η.")

    if verbose == True:
        print(f"Frequency sweep rate, a: {a} (Rad/s)/s = {a*60/(2*np.pi)} (RPM/s)")

    eta = get_eta(
                EO = EO, Q = Q, a = a, 
                #   f_n = f_n,
                Omega_n = Omega_n
                )
    
    A = get_A(eta=eta)
    f = get_f(eta=eta)
    zeta = get_zeta(eta=eta)
    # zeta = 1/(2*Q)

    # v = (Q/zeta) * ( # Option 1
    #                     # Defined just under Equation (16) in Reference [1]
    #             (
    #                 1 - (
    #                     (Omega_arr/(f/Q + 1)*Omega_n)**2
    #                     )
    #             )
    #             /
    #             (Omega_arr/(f/Q + 1)*Omega_n)
    #             )
    v = (Q/zeta) * ( # Option 2
                        # Defined just under Equation (16) in Reference [1]
                # (1 - ((Omega_arr/((f/Q + 1)*Omega_n))**2))/
                # (Omega_arr/((f/Q + 1)*Omega_n))
                (
                    1 - (
                        (Omega_arr/((f/Q + 1)*Omega_n))**2
                        )
                )
                /
                (Omega_arr/((f/Q + 1)*Omega_n))
                )

    numerator_term1 = A * A_max
    numerator_term2 = v*np.cos(EO*theta_sensor + phi_0) + np.sin(EO*theta_sensor + phi_0) # Not sure if the EO should be calculated to be an array of length Omega_arr?
    
    # denominator_term1 = Omega_arr/(f/Q + 1)*Omega_n #Option 1
    #                                                 # Not sure if its .../((f/Q + 1)*speed_n) or if its .../(f/Q + 1)*speed_n
    denominator_term1 = Omega_arr/( # Option 2
                                    (f/Q + 1)*Omega_n) # Not sure if its .../((f/Q + 1)*speed_n) or if its .../(f/Q + 1)*speed_n
    denominator_term2 = (v**2 + 1)

    predicted_tip_deflections = (numerator_term1/denominator_term1) * (numerator_term2/denominator_term2) + C
    
    return predicted_tip_deflections

def SDoF_loss_multiple_probes_sweep(
    model_params : np.ndarray,
    tip_deflections_set : List[np.ndarray],
    EO : int, 
    theta_sensor_set : List[float],

    # Variables to calculate the frequency sweep rate, a:
    Omega_arr: np.ndarray,
    time_arr: np.ndarray,

    # Omega_n: float, # now optimising for Omega_n so its included in model_params

    amplitude_scaling_factor : float = 1,
    verbose: bool = False

    ) -> np.ndarray:
    """ 
    Loss function for fitting SDoF sweep model to multiple probes.

    This function fits the SDoF parameters to multiple probes' data.

    Parameters
    ----------
    model_params : np.ndarray
        An array of the following model parameters: [phi_0, Q, A_max, C, Omega_n] where
        - phi_0 : float
            Initial phase of the excitation force (rad).
        - Q : float
            Quality factor (dimensionless).
        - A_max : float
            Resonant maximum amplitude of the blade vibration.
        - C : float
            Constant vibration offset term used when calculating the predicted tip deflections.
        - Omega_n : float
            The rotating speed at resonance (in rad/s).
    tip_deflections_set : list of np.ndarray
        Measured tip deflections for each probe.
    EO : int
        Engine order.
    theta_sensor_set : list of float
        Sensor angles (rad) for each probe.
    Omega_arr : np.ndarray
        Excitation speeds (rad/s).
    time_arr : np.ndarray
        Corresponding times (s).
    amplitude_scaling_factor : float, optional
        Exponent weight for measured amplitudes. Default is 1.
    verbose : bool, optional
        If True, prints progress for each probe.

    Returns
    -------
    float
        Total loss (sum of weighted squared errors).

    References
    ----------
    [1] F. Zhi et al., `Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate`,
    Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.
    """
    # omega_n, ln_zeta, delta_st, phi_0, *correction_factors = model_params


    # EO, theta_sensor, phi_0, Q, a, f_n, A_max, C, speed, speed_n = model_params
    # EO, theta_sensor, phi_0, Q, a, f_n, A_max, C = model_params
    # a = get_frequency_sweep_rate_a(
    #                             speed_at_start = Omega_arr[0], speed_at_end = Omega_arr[-1],
    #                             time_at_start = time_arr[0], time_at_end = time_arr[-1]
    #                             )
    phi_0, Q, A_max, C, Omega_n = model_params

    # zeta = np.exp(ln_zeta)


    error = 0
    for i_probe, arr_tip_deflections in enumerate(tip_deflections_set):    
        theta_sensor = theta_sensor_set[i_probe]
        predicted_tip_deflections = predict_sdof_samples_sweep(
            # omega_n, zeta, delta_st, EO, theta_sensor, phi_0, arr_omega
            EO = EO,
            theta_sensor = theta_sensor,
            phi_0 = phi_0,
            Q = Q,
            # a = a,
            # f_n = f_n,
            A_max = A_max,
            C = C,
            Omega_arr = Omega_arr,
            time_arr = time_arr,

            Omega_n = Omega_n, # now optimising for Omega_n

            # verbose = verbose
            )   

        # z_median = correction_factors[i_probe*2]
        # z_max = correction_factors[i_probe*2+1]
        # arr_tip_deflection_corrections = predict_sdof_samples_sweep(
        #     arr_omega, z_median, z_max
        # )
        # arr_tip_deflections_corrected = (
        #     arr_tip_deflections
        #     + arr_tip_deflection_corrections
        # )

        # arr_tip_deflection = predicted_tip_deflections
        error += np.sum(
            np.abs(arr_tip_deflections)**amplitude_scaling_factor
            *(
                arr_tip_deflections
                - predicted_tip_deflections
            )**2
        )
    return error


# def perform_SDoF_fit(
#     df_blade : pd.DataFrame,
#     n_start : int,
#     n_end : int,
#     EOs : List[int] = np.arange(1, 50),
#     # delta_st_max : int = 10,
#     verbose : bool = False,
#     probe_column_label_suffix : str = "_filt"
# ) -> Dict[str, float]:
#     """This function receives a blade tip deflection DataFrame, and returns 
#     the SDoF fit model parameters after fitting.

#     Args:
#         df_blade (pd.DataFrame): The blade tip deflection DataFrame.
#         n_start (int): The starting revolution number of the resonance 
#             you want to fit.
#         n_end (int): The ending revolution number of the resonance 
#             you want to fit.
#         EOs (List[int], optional): The list of EOs to search for. Defaults 
#             to np.arange(1, 20).
#         delta_st_max (int, optional): The maximum static deflection within our optimization 
#             bounds. Defaults to 10.
#         verbose (bool, optional): Whether to print the progress. Defaults to False.

#     Returns:
#         Dict[str, float]: The fitted model parameters.
#     """
#     df_resonance_window = df_blade.query(f"n >= {n_start} and n <= {n_end}")
#     measured_tip_deflection_signals = [
#         col 
#         for col in df_resonance_window
#         if col.endswith(probe_column_label_suffix)
#     ]
#     PROBE_COUNT = len(measured_tip_deflection_signals)
#     eo_solutions = []
#     for EO in EOs:
#         if verbose:
#             print("NOW SOLVING FOR EO = ", EO, " of ", EOs)
#         omega_n_min = df_resonance_window["Omega"].min() * EO
#         omega_n_max = df_resonance_window["Omega"].max() * EO
#         ln_zeta_min = np.log(0.0001)
#         ln_zeta_max = np.log(0.3)
#         delta_st_min = 0
#         phi_0_min = 0
#         phi_0_max = 2*np.pi
#         bounds = [
#             (omega_n_min, omega_n_max),
#             (ln_zeta_min, ln_zeta_max),
#             (delta_st_min, delta_st_max),
#             (phi_0_min, phi_0_max),
#         ]
#         tip_deflections_set = []
#         theta_sensor_set = []
#         for i_probe in range(PROBE_COUNT):
#             z_max = df_resonance_window[f"x_p{i_probe+1}"+probe_column_label_suffix].abs().max()
#             z_min = -z_max
#             bounds.extend(
#                 [
#                     (z_min, z_max),
#                     (z_min, z_max)
#                 ]
#             )
#             tip_deflections_set.append(
#                 df_resonance_window[f"x_p{i_probe+1}"+probe_column_label_suffix].values
#             )
#             theta_sensor_set.append(
#                 df_resonance_window[f"AoA_p{i_probe+1}"].median()
#             )
#         multiple_probes_solution = differential_evolution(
#             func = SDoF_loss_multiple_probes,
#             bounds=bounds,
#             args=(
#                 tip_deflections_set,
#                 df_resonance_window['Omega'].values,
#                 EO,
#                 theta_sensor_set,
#                 2
#             ),
#             seed=42
#         )
#         eo_solutions.append(multiple_probes_solution)
#     best_EO_arg = np.argmin([solution.fun for solution in eo_solutions])
#     best_EO = EOs[best_EO_arg]
#     best_solution = eo_solutions[best_EO_arg]
#     return {
#         "omega_n" : best_solution.x[0] / (2*np.pi),
#         "zeta" : np.exp(best_solution.x[1]),
#         "delta_st" : best_solution.x[2],
#         "phi_0" : best_solution.x[3],
#         "EO" : best_EO,
#     }