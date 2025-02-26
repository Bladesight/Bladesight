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

    for col in all_aoa_columns:
        df[col + "_norm"] = df[col].mean() - df[col]
        deflection_col_name = col.replace("AoA", "x")
        df[deflection_col_name] = blade_radius * df[col + "_norm"]
        poly = np.polyfit(df['Omega'], df[deflection_col_name], poly_order)
        df[deflection_col_name] = df[deflection_col_name] - np.polyval(poly, df['Omega'])
        
        if filter_function is not None and filter_kwargs is not None: # Filter tip deflections and denoted by columns with "_filt subfix"
            df[deflection_col_name + '_filt'] = filter_function(df[deflection_col_name].values, **filter_kwargs)
    
    if filter_function is not None and filter_kwargs is not None:
        x_matrix = df[[col for col in df.columns if col.endswith("_filt")]].to_numpy()
    else:
        x_matrix = df[[col for col in df.columns if col.startswith('x_p') and not col.endswith('_filt')]].to_numpy()
    
    df["pk-pk"] = x_matrix.max(axis=1) - x_matrix.min(axis=1) # If a filter function is supplied, the pk-pk values will be calculated from the filtered deflections

    return df


def get_blade_tip_deflections_from_AoAs_multi_col_filtering(
    df_rotor_blade_AoAs: pd.DataFrame,
    blade_radius: float,
    poly_order: int = 11,
    filter_function: Optional[Callable] = None,
    filter_kwargs: Optional[dict] = None,
    apply_filter_to_all_columns_at_once: bool = False,
    verbose: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Detrends blade tip deflections from AoA data, then optionally applies a filter.
    The filter can operate in one of two modes:

    1. Single-Column Mode:
        The filter function is applied separately to each deflection column.
        (e.g., a simple lowpass filter that doesn't require other columns)

    2. Multi-Column Mode:
        The filter function is called once with a 2D array of shape
        (n_samples, n_deflection_columns). This is useful for multi-signal
        methods like 'hankel_denoising' or PCA-based filters that jointly
        consider all columns.

    Parameters
    ----------
    df_rotor_blade_AoAs : pd.DataFrame
        Contains AoAs of each probe, typically with columns named 'AoA_pN'.
    blade_radius : float
        Radius of the blade (e.g., in microns).
    poly_order : int, optional
        Polynomial order for detrending, by default 11.
    filter_function : Optional[Callable], optional
        A user-provided function to filter the deflections. If None, no filter is applied.
        - For single-column mode, the function should accept a 1D NumPy array and return a 1D array.
        - For multi-column mode, it should accept a 2D NumPy array of shape
            (n_samples, n_columns) and return a 2D array of the same shape.
    filter_kwargs : Optional[dict], optional
        Named arguments passed to the filter function.
    apply_filter_to_all_columns_at_once : bool, default=False
        If False, each deflection column is passed to 'filter_function' separately.
        If True, all deflection columns are stacked into a 2D array and passed at once.
    verbose : bool, default=False
        Prints extra info if True.

    Returns
    -------
    pd.DataFrame
        A copy of the input with added columns:
        - AoA_pN_norm: AoA columns shifted by mean
        - x_pN: tip deflection signals (detrended)
        - x_pN_filt: filtered signals (if 'filter_function' is provided)
        - pk-pk: peak-to-peak deflection across columns (filtered if available)

    Example
    -------
    # 1) Column-by-column filter usage:
    def example_filter(signal_1d, window_size=5):
        return scisig.medfilt(signal_1d, kernel_size=window_size)

    df_filtered = get_blade_tip_deflections_from_AoAs(
        df_rotor_blade_AoAs,
        blade_radius=100.0,
        filter_function=example_filter,
        filter_kwargs={'window_size': 7},
        apply_filter_to_all_columns_at_once=False
    )

    # 2) Multi-column filtering usage (e.g., PCA or hankel_denoising):
    def example_multi_signal_filter(matrix_2d, n_components=1):
        # matrix_2d: shape (n_samples, n_columns)
        # return shape must match input
        # ...
        return matrix_2d  # placeholder for demonstration

    df_filtered_multi = get_blade_tip_deflections_from_AoAs(
        df_rotor_blade_AoAs,
        blade_radius=100.0,
        filter_function=example_multi_signal_filter,
        filter_kwargs={'n_components': 1},
        apply_filter_to_all_columns_at_once=True
    )
    """

    if filter_kwargs is None:
        filter_kwargs = {}

    df = df_rotor_blade_AoAs.copy(deep=True)
    all_aoa_columns = [col for col in df.columns if col.startswith("AoA_p")]
    if verbose:
        print(f"Detected AoA columns: {all_aoa_columns}")

    # 1) Normalize and detrend each AoA column -> storing results in x_pN
    deflection_cols = []
    for col in all_aoa_columns:
        # Shift by mean
        norm_col = col + "_norm"
        df[norm_col] = df[col].mean() - df[col]

        # Scale to tip deflection
        deflection_col_name = col.replace("AoA", "x")
        df[deflection_col_name] = blade_radius * df[norm_col]

        # Polynomial detrend
        poly = np.polyfit(df["Omega"], df[deflection_col_name], poly_order)
        df[deflection_col_name] -= np.polyval(poly, df["Omega"])
        deflection_cols.append(deflection_col_name)

    # 2) Apply filter if provided
    # print(f"{apply_filter_to_all_columns_at_once=}")
    if filter_function is not None:
        if apply_filter_to_all_columns_at_once == False:
            # ---- Single-Column Mode ----
            # Filter each column separately
            for col in deflection_cols:
                filtered_data = filter_function(df[col].values, **filter_kwargs)
                df[col + "_filt"] = filtered_data
            # Use newly filtered columns for peak-to-peak
            x_matrix = df[[col + "_filt" for col in deflection_cols]].to_numpy()

        if apply_filter_to_all_columns_at_once == True:
            # ---- Multi-Column Mode ----
            # Stack columns => shape (n_rows, n_cols)
            unfiltered_matrix = df[deflection_cols].to_numpy().reshape(len(deflection_cols), -1)
            # print("unfiltered_matrix.shape", unfiltered_matrix.shape)
            # filtered_matrix = filter_function(unfiltered_matrix, **filter_kwargs)
            # if filtered_matrix.shape != unfiltered_matrix.shape:
            #     raise ValueError("Filter did not return a matrix matching input shape.")

            # Store each filtered column
            for i, col in enumerate(deflection_cols):
                filter_kwargs["target_signal_index"] = i
                filtered_signal = filter_function(unfiltered_matrix, **filter_kwargs)
                # df[col + "_filt"] = filtered_matrix[:, i]
                df[col + "_filt"] = filtered_signal
                # df[col + "_filt"] = filter_function(unfiltered_matrix, **filter_kwargs)

            # Use newly filtered columns for peak-to-peak
            x_matrix = df[[col + "_filt" for col in deflection_cols]].to_numpy()
    else:
        # No filter, use unfiltered columns for peak-to-peak
        x_matrix = df[deflection_cols].to_numpy()

    # 3) Compute peak-to-peak
    # Each row => measure range across columns
    df["pk-pk"] = x_matrix.max(axis=1) - x_matrix.min(axis=1)

    if verbose:
        filter_mode = "all columns at once" if apply_filter_to_all_columns_at_once else "each column separately"
        print(f"Filtering mode: {filter_mode}")
        if filter_function is not None:
            print("Filtering was applied.")
        else:
            print("No filtering was applied.")
        print(f"Final columns in df: {list(df.columns)}")

    return df