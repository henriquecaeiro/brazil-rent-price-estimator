# -*- coding: utf-8 -*-
"""Module for loading trained models and data."""

import joblib
from pathlib import Path
import sys
import pandas as pd
from typing import List

# Add the api directory to the system path to ensure correct module imports.
API_DIR = Path(__file__).resolve().parent
if str(API_DIR) not in sys.path:
    sys.path.append(str(API_DIR))

# Import the function for creating extra features.
try:
    from features import create_extra_features
except ImportError:
    print("WARNING: Failed to import 'create_extra_features' from features.py.")

# Define the directory where the models are stored.
MODELS_DIR = API_DIR.parent / "models"

def load_model(filename: str):
    """Load a trained model from a file.

    Args:
        filename (str): The name of the model file.

    Returns:
        The loaded model or None if an error occurs.
    """
    model_path = MODELS_DIR / filename
    print(f"Attempting to load model from: {model_path}")

    if not model_path.exists():
        print(f"!!! ERROR: Model file not found at: {model_path}")
        return None
    
    try:
        model = joblib.load(model_path)
        print(f"Model '{filename}' loaded successfully.")
        return model
    except Exception as e:
        print(f"!!! ERROR loading model '{filename}':")
        print(e)
        return None

def _resolve_csv_path() -> Path:
    """Resolve the path to the CSV file in the data/work directory."""
    here = Path(__file__).resolve().parent
    data_dir = (here / ".." / "data" / "work").resolve()
    # Search for common CSV file patterns.
    for pat in ("*.csv", "*train*.csv", "*dataset*.csv"):
        found = list(data_dir.glob(pat))
        if found:
            return found[0]
    raise FileNotFoundError("CSV path not found. Set DATA_CSV env var.")

# Define the path to the CSV file.
CSV_PATH = _resolve_csv_path()

def _load_df() -> pd.DataFrame:
    """Load the dataframe from the CSV file and perform basic cleaning."""
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Normalize string columns for deduplication.
    for c in ("address", "district", "type"):
        if c in df.columns:
            s = df[c]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = s.map(lambda v: (str(v).strip()) if pd.notna(v) else None)
            df[c] = s.astype("object")
    return df

def get_unique_values(column: str) -> List[str]:
    """Get the unique values for a given column in the dataframe.

    Args:
        column (str): The name of the column.

    Returns:
        List[str]: A sorted list of unique values.
    """
    df = _load_df()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not in CSV. Available: {list(df.columns)}")

    s = df[column]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]

    # Convert to string, remove NaN and empty values.
    s = s.dropna().map(lambda v: str(v).strip())
    s = s[s.str.len() > 0]

    # Get unique values and sort them.
    return sorted(pd.unique(s).tolist())

def refresh_dataset_cache() -> None:
    """Clear the cache of the _load_df function."""
    _load_df().cache_clear()
