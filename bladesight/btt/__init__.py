from typing import List
import pandas as pd

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

def get_rotor_blade_AoAs(
    df_encoder_zero_crossings : pd.DataFrame,
    prox_probe_toas : List[pd.DataFrame],
    probe_spacings : List[float],
    B : int,
    encoder_sections : int = 1
) -> List[pd.DataFrame]:
    """This function converts the raw time stamps, both the OPR zero-crossing
    times and he proximity probe ToAs, and returns a DataFrame for each 
    blade containing the ToA and AoA values for each blade over all the
    proximity probes.

    Args:
        df_encoder_zero_crossings (pd.DataFrame): A DataFrame containing the
            zero-crossing times in its first column. If these zero-crossing times
            were recorded using an MPR shaft encoder, encoder_sections should be
            greater than 1.
        prox_probe_toas (List[pd.DataFrame]): A list of DataFrames
            where each DataFrame contains the ToAs of a single
            blade from a proximity probe.
        probe_spacings (List[float]): The relative spacing between
            the first probe and every other probe. There are one
            less value in this list than in prox_probe_toas.
        B (int): The number of blades.
        encoder_sections (int, optional): The number of sections
            the encoder is divided into. Defaults to 1.

    Returns:
        List[pd.DataFrame]: A list of DataFrames where each DataFrame
            contains the ToAs and AoAs of a single blade over all
            the proximity probes.
    """
    blade_dfs_recombined = []

    if encoder_sections == 1:
        for df_prox_toas in prox_probe_toas:
            df_prox = transform_ToAs_to_AoAs(
                df_encoder_zero_crossings, 
                df_prox_toas, 
            )
            
            blade_dfs_recombined.append(
                pivot_blade_AoAs_along_revolutions(
                    transform_prox_AoAs_to_blade_AoAs(
                        df_prox, 
                        B
                    )
                )
            )
    elif encoder_sections > 1:
        for df_prox_toas in prox_probe_toas:
            df_prox = transform_ToAs_to_AoAs_mpr(
                df_encoder_zero_crossings, 
                df_prox_toas, 
                encoder_sections
            )
            
            blade_dfs_recombined.append(
                pivot_blade_AoAs_along_revolutions(
                    transform_prox_AoAs_to_blade_AoAs(
                        df_prox, 
                        B
                    )
                )
            )
    
    rotor_AoA_dfs = assemble_rotor_AoA_dfs(
        prox_aligned_dfs=blade_dfs_recombined,
        probe_spacing=probe_spacings
    )
    return rotor_AoA_dfs