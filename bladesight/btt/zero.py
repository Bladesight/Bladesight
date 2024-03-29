import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def get_blade_tip_deflections_from_AoAs(
    df_rotor_blade_AoAs : pd.DataFrame,
    blade_radius : float,
    poly_order : int = 11,
    filter_order : int = 2,
    filter_cutoff : float = 0.3
) -> pd.DataFrame:
    """This function performs the following operations:
        1. Normalizes the AoAs of each probe.
        2. Scales the AoAs to tip deflections.
        3. Detrends the tip deflections using a polynomial fit.
        4. Filters the tip deflections using a Butterworth filter.
        5. Calculates the peak-to-peak tip deflection.

    Args:
        df_rotor_blade_AoAs (pd.DataFrame): The DataFrame containing the AoAs of each 
            probe. This is an item from the list returned by the 
            function `get_rotor_blade_AoAs`.
        blade_radius (float): The radius of the blade in microns.
        poly_order (int, optional): The polynomial order to use for the detrending
            algorithm . Defaults to 11.
        filter_order (int, optional): The order of the butterworth filter. Defaults to 2.
        filter_cutoff (float, optional): The butterworth filter cutoff 
            frequency. Defaults to 0.3.

    Returns:
        pd.DataFrame: The DataFrame containing the detrended and filtered 
            tip deflections. This DataFrame also contains the peak-to-peak
            tip deflection.
    """
    df = df_rotor_blade_AoAs.copy(deep=True)
    all_aoa_columns = [
        col_name 
        for col_name 
        in df.columns 
        if col_name.startswith("AoA_p")
    ]
    butterworth_filter = butter(N=filter_order, Wn=filter_cutoff)
    for col in all_aoa_columns:
        df[col + "_norm"] = df[col].mean() - df[col]
        deflection_col_name = col.replace("AoA", "x")
        df[deflection_col_name] = blade_radius * df[col + "_norm"]
        poly = np.polyfit(df['Omega'], df[deflection_col_name], poly_order)
        df[deflection_col_name] = df[deflection_col_name] - np.polyval(poly, df['Omega'])
        df[deflection_col_name + '_filt'] = filtfilt(*butterworth_filter, df[deflection_col_name])
    x_matrix = df[[col for col in df.columns if col.endswith("_filt")]].to_numpy()
    df["pk-pk"] = x_matrix.max(axis=1) - x_matrix.min(axis=1)
    return df
