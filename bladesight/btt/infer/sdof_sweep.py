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
from typing import List, Dict, Optional


from .sdof import get_phi

# ------------------ Frequency Sweep Parameter Method ------------------ #
def get_eta(
            EO: int, 
            Q: float, a: float, 
            # f_n: float,
            speed_at_resonance:float
            ) -> float:
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
    f_n : float
        Natural frequency (in Hz or consistent units).
    speed_at_resonance : float
        The rotating speed at resonance (in radians/second). 
        This is denoted by Ωₙ in Reference [1] but this notation is not used here to avoid confusion with the omega_n term in sdof.py that denotes the blade vibration natural frequency.

    Returns
    -------
    float
        The computed non-dimensional Frequency sweep parameter η.

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.
    """
    # return EO*((Q**2) * a)/(60*(f_n**2)) # Equation 11 in Reference [1], NOT using Omega_n as its confusing with existing bladesight notation in sdof.py
    return ((Q**2) * a)/(60*EO*(speed_at_resonance**2)) # Equation 11 in Reference [1], NOT using f_n as its just an extra parameter to pass in unnecessarily

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
        The rotating speed at the start of the frequency sweep (in radians/second).
    speed_at_end : float
        The rotating speed at the end of the frequency sweep (in radians/second).
    time_at_start : float
        The time at the start of the frequency sweep (in seconds).
    time_at_end : float
        The time at the end of the frequency sweep (in seconds).

    Returns:
    --------
    float
        The frequency sweep rate, a (in radians/second^2).

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.

    Notes:
    ------
    In Section 3.2 of Reference [1], the frequency sweep rate should be smaller than 216*(f_n**2)*ζ**2.    
    """
    return (speed_at_end - speed_at_start) / (time_at_end - time_at_start)

def predict_SDoF_sweep_samples(
    EO : int,
    theta_sensor : float,
    phi_0 : float,
    # Reference [1] parameters:
    Q: float,
    # a: float,
    # f_n: float,
    A_max: float,
    C: float,
    speed_at_resonance: float, # Moved Omega_n up here so it can be optimised


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
    speed_at_resonance : float
        The rotating speed at resonance (in radians/second). 
        This is denoted by Ωₙ in Reference [1] but this notation is not used here to avoid confusion with the omega_n term in sdof.py that denotes the blade vibration natural frequency.
    Omega_arr : np.ndarray
        Array of excitation frequencies (in radians/second).
    time_arr : np.ndarray
        Array of corresponding times (seconds).
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
                EO = EO, 
                Q = Q, 
                a = a, 
                #   f_n = f_n,
                speed_at_resonance = speed_at_resonance
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
                        (Omega_arr/((f/Q + 1)*speed_at_resonance))**2
                        )
                )
                /
                (Omega_arr/((f/Q + 1)*speed_at_resonance))
                )

    numerator_term1 = A * A_max
    numerator_term2 = v*np.cos(EO*theta_sensor + phi_0) + np.sin(EO*theta_sensor + phi_0) # Not sure if the EO should be calculated to be an array of length Omega_arr?
    
    # denominator_term1 = Omega_arr/(f/Q + 1)*Omega_n #Option 1
    #                                                 # Not sure if its .../((f/Q + 1)*speed_n) or if its .../(f/Q + 1)*speed_n
    denominator_term1 = Omega_arr/( # Option 2
                                    (f/Q + 1)*speed_at_resonance) # Not sure if its .../((f/Q + 1)*speed_n) or if its .../(f/Q + 1)*speed_n
    denominator_term2 = (v**2 + 1)

    predicted_tip_deflections = (numerator_term1/denominator_term1) * (numerator_term2/denominator_term2) + C
    
    return predicted_tip_deflections

def SDoF_sweep_loss_multiple_probes(
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
        - speed_at_resonance : float
            The rotating speed at resonance (in radians/second). 
            This is denoted by Ωₙ in Reference [1] but this notation is not used here to avoid confusion with the omega_n term in sdof.py that denotes the blade vibration natural frequency.
    tip_deflections_set : list of np.ndarray
        Measured tip deflections for each probe.
    EO : int
        Engine order.
    theta_sensor_set : list of float
        Sensor angles (rad) for each probe.
    Omega_arr : np.ndarray
        Excitation speeds (radians/second).
    time_arr : np.ndarray
        Corresponding times (seconds).
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
    phi_0, Q, A_max, C, speed_at_resonance = model_params

    # zeta = np.exp(ln_zeta)


    error = 0
    for i_probe, arr_tip_deflections in enumerate(tip_deflections_set):    
        theta_sensor = theta_sensor_set[i_probe]
        predicted_tip_deflections = predict_SDoF_sweep_samples(
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

            speed_at_resonance = speed_at_resonance, # now optimising for speed_at_resonance so its included in model_params

            # verbose = verbose
            )   
        error += np.sum(
            np.abs(arr_tip_deflections)**amplitude_scaling_factor
            *(
                arr_tip_deflections
                - predicted_tip_deflections
            )**2
        )
    return error


def perform_SDoF_sweep_fit(
    df_blade : pd.DataFrame,
    n_start : int,
    n_end : int,
    EOs : Optional[List[int]] = np.arange(1, 20),
    # omega_n_bounds : Optional[List[float]] = [None, None],
    # zeta_bounds : Optional[List[float]] = [0.0001, 0.3],
    # delta_st_bounds : Optional[List[float]] = [0, 10],
    # phi_0_bounds : Optional[List[float]] = [0, 2*np.pi],

    phi_0_bounds : Optional[List[float]] = [0, 2*np.pi],
    Q_bounds : Optional[List[float]] = [1, 1000],
    A_max_bounds : Optional[List[float]] = [-1000, 1000],
    C_bounds : Optional[List[float]] = [-1000, 1000],
    speed_at_resonance_bounds : Optional[List[float]] = [None, None],



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



    # omega_n_bounds : List[float], optional
    #     [min, max] bounds for resonant speed Ωₙ (radians/second). If None, set per EO as
    #     (min(df_blade['Omega'])*EO, max(df_blade['Omega'])*EO).
    # zeta_bounds : list of float, optional
    #     [min, max] bounds for damping ratio ζ, default to: [0.0001, 0.3].
    # delta_st_bounds : list of float, optional
    #     [min, max] bounds for static deflection δₛₜ, , defaults to [0, 10].
    # phi_0_bounds : list of float, optional
    #     [min, max] bounds for phase offset φ₀ in radians, defaults to [0, 2π].

    
    phi_0_bounds : list of float, optional
        [min, max] bounds for phase offset φ₀ in radians, defaults to [0, 2π].
    Q_bounds : list of float, optional
        [min, max] bounds for quality factor Q, defaults to [1, 1000].
    A_max_bounds : list of float, optional
        [min, max] bounds for resonant maximum amplitude A_max, defaults to [-1000, 1000].
    C_bounds : list of float, optional
        [min, max] bounds for constant vibration offset C, defaults to [-1000, 1000].
    speed_at_resonance_bounds : list of float, optional
        [min, max] bounds for speed at resonance (Ωₙ), defaults to [None, None].
    
    signal_suffix : str, optional
        Suffix appended to deflection column names, defaults to "_filt".

        
    # amplitude_scaling_factor : float, optional
    #     Scaling factor for amplitude weighting, defaults to 1.


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
        # - "omega_n": Natural frequency (in Hz).
        # - "zeta": Damping ratio.
        # - "delta_st": Static deflection.
        # - "phi_0": Phase offset in radians.
        # - "EO": Engine order with minimum error.

        - "phi_0": Initial phase of the excitation force (in radians).
        - "Q": Quality factor.
        - "A_max": Resonant maximum amplitude.
        - "C": Constant vibration offset.
        - "speed_at_resonance": Speed at resonance (in radians/second).
        - "EO": Engine order with minimum error.
    """
    df_resonance_window = df_blade.query(f"n >= {n_start} and n <= {n_end}")
    measured_tip_deflection_signals = [
        col 
        for col in df_resonance_window
        if col.endswith("_filt")
    ]
    PROBE_COUNT = len(measured_tip_deflection_signals)
    EO_solutions = []
    for EO in EOs:
        if verbose:
            print("NOW SOLVING FOR EO = ", EO, " of ", EOs)
        
        if speed_at_resonance_bounds.__contains__(None):
            speed_at_resonance_bounds[0] = df_resonance_window["Omega"].min()
            speed_at_resonance_bounds[1] = df_resonance_window["Omega"].max()
        if speed_at_resonance_bounds.__contains__(None) == False:
            speed_at_resonance_bounds[0] = speed_at_resonance_bounds[0]
            speed_at_resonance_bounds[1] = speed_at_resonance_bounds[1]

        bounds = [
            # (omega_n_bounds[0], omega_n_bounds[1]),
            # (zeta_bounds[0], zeta_bounds[1]),
            # (delta_st_bounds[0], delta_st_bounds[1]),
            # (phi_0_bounds[0], phi_0_bounds[1]),
            (phi_0_bounds[0], phi_0_bounds[1]),
            (Q_bounds[0], Q_bounds[1]),
            (A_max_bounds[0], A_max_bounds[1]),
            (C_bounds[0], C_bounds[1]),
            (speed_at_resonance_bounds[0], speed_at_resonance_bounds[1]),
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
            func = SDoF_sweep_loss_multiple_probes,
            bounds=bounds,
            args=(
                tip_deflections_set,
                EO,
                theta_sensor_set,
                df_resonance_window['Omega'].values,
                df_resonance_window[f'ToA_p{i_probe+1}'].values,
                amplitude_scaling_factor,
            ),
            **differential_evolution_optimiser_kwargs
        )
        EO_solutions.append(multiple_probes_solution)
    
    # Select the best EO
    best_EO_arg = np.argmin([solution.fun for solution in EO_solutions])
    best_EO = EOs[best_EO_arg]
    best_solution = EO_solutions[best_EO_arg]
    return {
        # "omega_n" : best_solution.x[0] / (2*np.pi),
        # "zeta" : np.exp(best_solution.x[1]),
        # "delta_st" : best_solution.x[2],
        # "phi_0" : best_solution.x[3],
        # "EO" : best_EO,
        "phi_0" : best_solution.x[0],
        "Q" : best_solution.x[1],
        "A_max" : best_solution.x[2],
        "C" : best_solution.x[3],
        "speed_at_resonance" : best_solution.x[4],
        "EO" : best_EO,
    }