import numpy as np
import pandas as pd
from typing import List, Dict, Union

def cff_method_single_revolution(
    df_blade : pd.DataFrame,
    theta_sensor_set : List[float],
    EO : int,
    signal_suffix : str = "_filt" 
) -> pd.DataFrame:
    """ This function fits the CFF method for a resonance by 
    using a single revolution of data for each set of CFF parameters.

    Args:
        df_blade (pd.DataFrame): The dataframe containing the tip deflections.
        theta_sensor_set (List[float]): The sensor angles.
        EO (int): The Engine Order.
        signal_suffix (str, optional): The suffix of the tip deflection 
            signals. Defaults to "_filt".

    Returns:
        pd.DataFrame: A DataFrame containing the CFF parameters for 
            each shaft revolution. 
    """
    PROBE_COUNT = len(theta_sensor_set)
    tip_deflection_signals = [
        f"x_p{i_probe + 1}{signal_suffix}" 
        for i_probe in range(PROBE_COUNT)
    ]
    theta_sensors = np.array(theta_sensor_set)
    A = np.ones((PROBE_COUNT, 3))
    A[:, 0] = np.sin(theta_sensors * EO)
    A[:, 1] = np.cos(theta_sensors * EO)

    A_pinv = np.linalg.pinv(A) 
    B = A_pinv.dot(
        df_blade.loc[:, tip_deflection_signals].values.T
    ) 
    df_cff = pd.DataFrame(B.T, columns=["A", "B", "C"]) 
    df_cff["X"] = np.sqrt(df_cff["A"]**2 + df_cff["B"]**2)
    df_cff["phi"] = np.arctan2(df_cff["A"], df_cff["B"])
    df_cff["n"] = df_blade["n"].values
    df_predicted_targets = pd.DataFrame(
        A.dot(B).T, 
        columns=[
            col + "_pred" 
            for col 
            in tip_deflection_signals
        ]
    ) 
    df_cff = pd.concat([df_cff, df_predicted_targets], axis=1)
    return df_cff

def cff_method_multiple_revolutions(
    df_blade : pd.DataFrame,
    theta_sensor_set : List[float],
    EO : int,
    extra_revolutions : int,
    signal_suffix : str = "_filt" 
) -> pd.DataFrame:
    """ This function fits the CFF method for a resonance by
    using multiple revolutions of data for each set of CFF parameters.

    Args:
        df_blade (pd.DataFrame): The dataframe containing the tip deflections.
        theta_sensor_set (List[float]): The sensor angles.
        EO (int): The Engine Order.
        extra_revolutions (int): The number of revolutions to use for the fit.
        signal_suffix (str, optional): The suffix of the tip deflection 
            signals. Defaults to "_filt".

    Returns:
        pd.DataFrame: A DataFrame containing the CFF parameters for 
            each shaft revolution.
    """
    PROBE_COUNT = len(theta_sensor_set)
    tip_deflection_signals = [
        f"x_p{i_probe + 1}{signal_suffix}" 
        for i_probe in range(PROBE_COUNT)
    ]
    theta_sensors = np.array(theta_sensor_set)

    A = np.ones((PROBE_COUNT*(2*extra_revolutions+1), 3))
    arr_multiple_thetas = np.array(
        list(theta_sensors)*(2*extra_revolutions+1)
    )
    A[:, 0] = np.sin(arr_multiple_thetas * EO)
    A[:, 1] = np.cos(arr_multiple_thetas * EO)
    A_pinv = np.linalg.pinv(A)
    new_obs_rows = df_blade.shape[0] - 2*extra_revolutions
    X_multiple_revos = np.zeros(
        (
            new_obs_rows, 
            PROBE_COUNT*(2*extra_revolutions+1)
        )
    )
    for n_revo in range(-extra_revolutions, extra_revolutions+1):
        for i_probe in range(PROBE_COUNT):
            mat_aoas_start = extra_revolutions + n_revo
            mat_aoas_end = mat_aoas_start + new_obs_rows
            i_col = i_probe + n_revo*PROBE_COUNT + extra_revolutions*PROBE_COUNT
            X_multiple_revos[:,i_col] = (
                df_blade.iloc[mat_aoas_start:mat_aoas_end][tip_deflection_signals[i_probe]]
            )
    B = A_pinv.dot(X_multiple_revos.T)
    B_full = np.zeros((df_blade.shape[0], 3))
    B_full[extra_revolutions:-extra_revolutions, :] = B.T
    B_full[:extra_revolutions, :] = B_full[extra_revolutions, :]
    B_full[-extra_revolutions:, :] = B_full[-extra_revolutions-1, :]

    df_cff = pd.DataFrame(B_full, columns=["A", "B", "C"])
    df_cff["X"] = np.sqrt(df_cff["A"]**2 + df_cff["B"]**2)
    df_cff["phi"] = np.arctan2(df_cff["A"], df_cff["B"])
    df_cff["n"] = df_blade["n"].values
    target_matrix = (A.dot(B_full.T)).T
    predicted_deflections = target_matrix[:, extra_revolutions*PROBE_COUNT:(extra_revolutions+1)*PROBE_COUNT] 
    df_predicted_targets = pd.DataFrame(
        predicted_deflections, 
        columns=[
            col + "_pred" 
            for col 
            in tip_deflection_signals
        ]
    )
    df_cff = pd.concat([df_cff, df_predicted_targets], axis=1)
    return df_cff

def perform_CFF_fit(
    df_blade : pd.DataFrame,
    n_start : int,
    n_end : int,
    EOs : List[int] = np.arange(1, 20),
    extra_revolutions : int = 1
) -> Dict[str, Union[pd.DataFrame, int]]:
    """ This function performs the CFF method fit for a resonance. The function
    iterates over EOs and selects the EO that gives the lowest sum of squared
    errors between the measured and predicted tip deflections.

    Args:
        df_blade (pd.DataFrame): The dataframe containing the tip deflections.
        n_start (int): The start revolution number.
        n_end (int): The end revolution number.
        EOs (List[int], optional): The EOs to consider for this resonance. Defaults 
            to np.arange(1, 20).
        extra_revolutions (int, optional): How many extra revolutions to use for 
            the fit. Defaults to 1.

    Returns:
        Dict[str, Union[pd.DataFrame, int]]: A dictionary containing the CFF 
            parameters and the selected EO.
    """
    PROBE_COUNT = len(
        [
            col 
            for col in df_blade.columns
            if col.endswith("_filt")
        ]
    )
    theta_sensor_set = [
        df_blade[f"AoA_p{i_probe + 1}"].median()
        for i_probe in range(PROBE_COUNT)
    ]
    errors = []
    df_resonance_window = df_blade.query(f"n >= {n_start} and n <= {n_end}")
    for EO in EOs:
        df_cff_params = cff_method_multiple_revolutions(
            df_resonance_window,
            theta_sensor_set,
            EO,
            extra_revolutions
        )
        error = 0
        for i_probe in range(PROBE_COUNT):
            error += np.sum(
                (
                    df_cff_params[f"x_p{i_probe+1}_filt_pred"].values 
                    - df_resonance_window[f"x_p{i_probe+1}_filt"].values
                )**2
            )
        errors.append(error)
    EO = EOs[np.argmin(errors)]
    df_cff_params = cff_method_multiple_revolutions(
        df_resonance_window,
        theta_sensor_set,
        EO,
        extra_revolutions
    )
    return {
        "df_cff_params" : df_cff_params,
        "EO" : EO
    }