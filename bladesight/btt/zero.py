import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

from typing import Optional, Callable, Dict

def get_blade_tip_deflections_from_AoAs(
    df_rotor_blade_AoAs : pd.DataFrame,
    blade_radius : float,
    poly_order : int = 11,
    filter_function: Optional[Callable] = None,
    filter_kwargs: Optional[dict] = None,
    verbose: Optional[bool] = False,
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
        filter_function (Optional[Callable], optional): 
            The filter function to be applied to the tip deflections. If None, no filtering is applied. 
            Note that the filter function needs to only return the filtered signal that is the same shape and analogous to the unfiltered signal.
        filter_kwargs : Optional[dict], default=None
            The arguments to be passed to the filter function. If None, no filtering filter arguments are applied and use None when you dont want to filter.
        verbose : Optional[bool], default=False
            A flag to enable verbose output.
    Returns:
        pd.DataFrame: The DataFrame containing the detrended and filtered 
            tip deflections. This DataFrame also contains the peak-to-peak
            tip deflection.

    Example Usage:
        def example_filter(signal, kernel_size):
            return scisig.medfilt(signal, kernel_size=kernel_size)

        df_rotor_blade_AoAs = pd.DataFrame({
            'Omega': np.linspace(0, 10, 100),
            'AoA_p1': np.random.randn(100),
            'AoA_p2': np.random.randn(100)
        })

        blade_radius = 100.0
        filter_function = example_filter
        filter_args = {'kernel_size': 5}

        df_filtered = get_blade_tip_deflections_from_AoAs_kernel(
            df_rotor_blade_AoAs,
            blade_radius,
            filter_function,
            filter_args,
            poly_order=11,
            filter_deflections_bool=True,
            verbose=True
        )

        print(df_filtered.head())
    """
    if filter_kwargs is None:
        filter_kwargs = {}

    df = df_rotor_blade_AoAs.copy(deep=True)
    all_aoa_columns = [
        col_name 
        for col_name 
        in df.columns 
        if col_name.startswith("AoA_p")
    ]
    # butterworth_filter = butter(N=filter_order, Wn=filter_cutoff)

    for col in all_aoa_columns:
        df[col + "_norm"] = df[col].mean() - df[col]
        deflection_col_name = col.replace("AoA", "x")
        df[deflection_col_name] = blade_radius * df[col + "_norm"]
        poly = np.polyfit(df['Omega'], df[deflection_col_name], poly_order)
        df[deflection_col_name] = df[deflection_col_name] - np.polyval(poly, df['Omega'])
        
        if filter_function is not None and filter_kwargs is not None:
            df[deflection_col_name + '_filt'] = filter_function(df[deflection_col_name].values, **filter_kwargs)
    
    if filter_function is not None and filter_kwargs is not None:
        x_matrix = df[[col for col in df.columns if col.endswith("_filt")]].to_numpy()
    else:
        x_matrix = df[[col for col in df.columns if col.startswith('x_p') and not col.endswith('_filt')]].to_numpy()
    
    df["pk-pk"] = x_matrix.max(axis=1) - x_matrix.min(axis=1) # If a filter function is supplied, the pk-pk values will be calculated from the filtered deflections

    return df
