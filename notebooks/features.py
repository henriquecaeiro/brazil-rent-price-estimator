import numpy as np

def create_extra_features(df):
    df_out = df.copy()
    df_out['area_log'] = np.log1p(df_out['area'])
    df_out['area_per_bedroom'] = df_out['area'] / np.clip(df_out['bedrooms'].astype(float), 1, None)
    df_out['district_type'] = df_out['district'].astype(str) + '_' + df_out['type'].astype(str)
    return df_out