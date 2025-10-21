# -*- coding: utf-8 -*-
"""Module for creating extra features for the model."""

import numpy as np

def create_extra_features(df):
    """Create extra features for the model.

    Args:
        df (pd.DataFrame): The input dataframe.

    Returns:
        pd.DataFrame: The dataframe with the extra features.
    """
    df_out = df.copy()
    # Create a log-transformed version of the 'area' feature to handle skewed data.
    df_out['area_log'] = np.log1p(df_out['area'])
    # Create a feature representing the area per bedroom.
    df_out['area_per_bedroom'] = df_out['area'] / np.clip(df_out['bedrooms'].astype(float), 1, None)
    # Create a combined feature of district and type.
    df_out['district_type'] = df_out['district'].astype(str) + '_' + df_out['type'].astype(str)
    return df_out
