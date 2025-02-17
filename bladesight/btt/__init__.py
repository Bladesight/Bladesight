from typing import List, Union, Optional
import pandas as pd
import polars as pl

from .aoa import (
    transform_ToAs_to_AoAs, 
    transform_ToAs_to_AoAs_mpr, 
    transform_prox_AoAs_to_blade_AoAs
)
from .align import pivot_blade_AoAs_along_revolutions, assemble_rotor_AoA_dfs
from .zero import get_blade_tip_deflections_from_AoAs

__all__ = [
    "get_rotor_blade_AoAs",
    "get_blade_tip_deflections_from_AoAs"
]

def verbose_print(verbose: bool, text: str):
    """
    Print the text if the verbose flag is True.

    Parameters
    ----------
    verbose : bool
        A flag indicating whether to print the text or not.
    text : str
        The f-string for the text to print.
    Returns
    -------
    None, only prints the text if the verbose flag is True.
    """
    if verbose:
        print(text)


def get_rotor_blade_AoAs(
    df_encoder : pd.DataFrame,
    prox_probe_toas : List[Union[pd.DataFrame, pl.DataFrame]],
    probe_spacings : List[float],
    B : int,
    is_mpr : bool = False,
    tramsform_prox_AoAs_to_blade_AoAs_kwargs : Optional[dict] = None
) -> List[pd.DataFrame]:
    """This function converts the raw time stamps, both the OPR zero-crossing
    times and he proximity probe ToAs, and returns a DataFrame for each 
    blade containing the ToA and AoA values for each blade over all the
    proximity probes.

    Args:
        df_encoder (pd.DataFrame): A DataFrame containing the
            zero-crossing times and corresponding info. This DataFrame can
            take on two values: 
                1) OPR encoder: The first column of the DataFrame should be
                the zero-crossing times of the OPR encoder.
                2) MPR encoder: The DataFrame MUST be the result of the
                bladesight.ias.calculate_mpr function.
        prox_probe_toas (List[pd.DataFrame]): A list of DataFrames
            where each DataFrame contains the ToAs of a single
            blade from a proximity probe.
        probe_spacings (List[float]): The relative spacing between
            the first probe and every other probe. There are one
            less value in this list than in prox_probe_toas.
        B (int): The number of blades.
        is_mpr (bool, optional): A flag to indicate if the encoder is
            an MPR encoder. Defaults to False.
        tramsform_prox_AoAs_to_blade_AoAs_kwargs (Optional[dict], optional):
            The keyword arguments to be passed to the 
            transform_prox_AoAs_to_blade_AoAs function. Defaults to None.

    Returns:
        List[pd.DataFrame]: A list of DataFrames where each DataFrame
            contains the ToAs and AoAs of a single blade over all
            the proximity probes.
    """
    blade_dfs_recombined = []

    if not is_mpr:
        for df_prox_toas in prox_probe_toas:
            df_prox = transform_ToAs_to_AoAs(
                df_encoder.to_frame() if isinstance(df_encoder, (pd.Series, pl.Series)) else df_encoder, 
                df_prox_toas.to_frame() if isinstance(df_prox_toas, (pd.Series, pl.Series)) else df_prox_toas
            )
            
            blade_dfs_recombined.append(
                pivot_blade_AoAs_along_revolutions(
                    transform_prox_AoAs_to_blade_AoAs(
                        df_prox = df_prox, 
                        B = B,
                        **tramsform_prox_AoAs_to_blade_AoAs_kwargs,
                    )
                )
            )
    else:
        for df_prox_toas in prox_probe_toas:
            try:
                df_prox = transform_ToAs_to_AoAs_mpr(
                    df_encoder.to_frame() if isinstance(df_encoder, (pd.Series, pl.Series)) else df_encoder, 
                    df_prox_toas.to_frame() if isinstance(df_prox_toas, (pd.Series, pl.Series)) else df_prox_toas
                )
            except AssertionError as e:
                print("It looks like you did not pass the correct MPR encoder DataFrame.")
                print("Please ensure you pass df_encoder as the result of the bladesight.ias.calculate_mpr function.")
                raise e

            blade_dfs_recombined.append(
                pivot_blade_AoAs_along_revolutions(
                    transform_prox_AoAs_to_blade_AoAs(
                        df_prox = df_prox, 
                        B = B,
                        **tramsform_prox_AoAs_to_blade_AoAs_kwargs,
                    )
                )
            )
    
    rotor_AoA_dfs = assemble_rotor_AoA_dfs(
        prox_aligned_dfs=blade_dfs_recombined,
        probe_spacing=probe_spacings
    )
    return rotor_AoA_dfs