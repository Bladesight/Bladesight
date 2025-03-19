"""
This module contains the functions proposed by:
[1] F. Zhi et al., `Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate`,
Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.

The notation in this module follows the one used in the reference [1].
"""
import numpy as np
from typing import List, Dict


from .sdof import get_phi

# # ------------------ Tradidional Single-parameter Method ------------------ # - i realised after typing this that this is just the existing SDoF method
# def get_H(
#         omega : np.ndarray,
#         omega_n : float, 
#         zeta: float,
#         delta_st: float
#     ) -> np.ndarray:
#     """
#     This function returns the vibration amplitude of 
#     the blade vibration.
    
#     x(ω) = 	δ_st / sqrt( (1 - r**2)**2 + (2*ζ*r)**2)

#     where:

#     r = ω/ω_0

#     Parameters
#     ----------
#     omega : np.ndarray
#         The excitation frequencies in rad/s.
#     omega_n : float
#         The natural frequency of the blade in rad/s.
#     zeta : float
#         The damping ratio of the blade vibration (dimensionless).
#     delta_st : float
#         The static deflection of the blade (typically in µm).

#     Returns
#     -------
#     np.ndarray
#         The vibration amplitude of the blade corresponding to each frequency in `omega`
#         (in the same units as `delta_st`).

#     Notes
#     -----
#     In reference [1], the amplitude term A_s is equivalent to the static deflection δ_st used here.
    
#     References
#     ----------
#     [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
#         Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.
#     """
#     r = omega / omega_n
#     return (
#         delta_st 
#         / np.sqrt(
#             (1 - r**2)**2 
#             + (2*zeta*r)**2
#         )
#     )

# def predict_sdof_single_probe(
#     omega_n : float,
#     zeta : float,
#     delta_st : float,
#     EO : int,
#     theta_sensor : float,
#     arr_omega : np.ndarray
# ) -> np.ndarray:
#     """
#     Predict blade tip deflections for a single probe using a single-degree-of-freedom (SDOF) model.

#     This function calculates the predicted tip deflection of a blade based on an SDOF model 
#     using a single probe measurement. The model utilizes the vibration amplitude computed 
#     by `get_H` and the phase angle computed by `get_phi` for excitation frequencies scaled 
#     by the engine order (EO). An alternative formulation is then applied involving an 
#     initial excitation phase (phi_0), assumed to be defined elsewhere in the code.

#     The predicted tip deflection is computed via:
    
#         r = arr_omega / omega_n
#         numerator = (1 - r**2) * cos(EO*theta_sensor + phi_0) + (2*zeta*r) * sin(EO*theta_sensor + phi_0)
#         denominator = (1 - r**2)**2 + (2*zeta*r)**2
#         predicted_tip_deflections = delta_st * (numerator / denominator)

#     Here, phi_0 is the initial phase of the excitation force.

#     Parameters
#     ----------
#     omega_n : float
#         Natural frequency of the blade in rad/s.
#     zeta : float
#         Damping ratio of the blade vibration (dimensionless).
#     delta_st : float
#         Static deflection of the blade (typically in µm).
#     EO : int
#         Engine order used to scale the excitation frequencies.
#     theta_sensor : float
#         Sensor angle in radians, representing the absolute circumferential position.
#     arr_omega : np.ndarray
#         Array of excitation frequencies in rad/s.

#     Returns
#     -------
#     np.ndarray
#         The predicted blade tip deflection values (in the same units as `delta_st`) 
#         corresponding to each frequency in `arr_omega`.

#     References
#     ----------
#     [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
#         Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.

#     Notes
#     -----
#     - The functions `get_H` and `get_phi` are used to compute the amplitude and phase, 
#     respectively, for the scaled excitation frequencies (arr_omega * EO).
#     - The variable `phi_0` must be defined in an outer scope; it represents the initial 
#     phase of the excitation force.
#     - It is assumed that the variable `omega` in the model formulation is equivalent 
#     to `arr_omega`.
#     """
#     # X = get_H(arr_omega*EO, omega_n, zeta, delta_st)
#     # phi = get_phi(arr_omega*EO, omega_n, zeta)
#     # predicted_tip_deflections = X * np.cos(theta_sensor * EO - phi)

#     r = omega / omega_n

#     numerator_term1 = (1 - r**2) * np.cos(EO*theta_sensor + phi_0) # phi_0 is the initial phase of the excitation force
#     numerator_term2 = (2*zeta*r) * np.sin(EO*theta_sensor + phi_0) # phi_0 is the initial phase of the excitation force

#     numerator = numerator_term1 + numerator_term2
#     denominator = (1 - r**2)**2 + (2*zeta*r)**2

#     predicted_tip_deflections = delta_st * numerator/denominator

#     return predicted_tip_deflections


# ------------------ Frequency Sweep Parameter Method ------------------ #
def get_eta(EO: int, Q: float, a: float, f_n: float) -> float:
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
        A scaling parameter (units dependent on application).
    f_n : float
        Natural frequency (in Hz or consistent units).

    Returns
    -------
    float
        The computed non-dimensional parameter η.

    References
    ----------
    [1] F. Zhi et al., "Error Revising of Blade Tip-Timing Parameter Identification Caused by Frequency Sweep Rate",
        Measurement, vol. 201, p. 111681, Sep. 2022, doi: 10.1016/j.measurement.2022.111681.
    """
    return EO*((Q**2) * a)/(60*f_n**2)


def get_A(eta: float) -> float:
    """
    Compute the fraction of the peak amplitude factor, A, based on the non-dimensional parameter η.

    Equation (13) in Reference [1]:
        A = 1 - exp(-2.86 * η^(-0.455))

    Parameters
    ----------
    eta : float
        The non-dimensional parameter η.

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
        f = 0.518 * η^(0.576)

    Parameters
    ----------
    eta : float
        The non-dimensional parameter η.

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
        ζ = 1 / A

    where A is computed by `get_A(eta)`.

    Parameters
    ----------
    eta : float
        The non-dimensional parameter η.

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


def predict_sdof_samples_sweep(
    omega_n : float,
    zeta : float,
    delta_st : float,
    EO : int,
    theta_sensor : float,
    phi_0 : float,
    arr_omega : np.ndarray
) -> np.ndarray:
    
    # X = get_X(arr_omega*EO, omega_n, zeta, delta_st)  
    # phi = get_phi(arr_omega*EO, omega_n, zeta)
    # predicted_tip_deflections = X * np.cos(theta_sensor * EO - phi + phi_0)

    eta = get_eta(EO = EO, Q = Q, a = a, f_n = f_n)
    A = get_A(eta=eta)
    f = get_f(eta=eta)
    zeta = get_zeta(eta=eta)

    numerator_term1 = A * A_max
    numerator_term2 = v*np.cos(EO*theta_sensor + phi_0) + np.sin(EO*theta_sensor + phi_0)
    
    v = Q/zeta * (
                (1 - ((speed/(f/Q + 1))*speed_n)**2)/
                (((speed/(f/Q + 1))*speed_n)**2)
                )



    denominator_term1 = speed/((f/Q + 1)*speed_n) # Not sure if its .../((f/Q + 1)*speed_n) or if its .../(f/Q + 1)*speed_n
    denominator_term2 = (v**2 + 1)

    predicted_tip_deflections = (numerator_term1/denominator_term1) * (numerator_term2/denominator_term2) + C
    
    
    return predicted_tip_deflections