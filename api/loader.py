# api/loader.py
import json
import joblib
from pathlib import Path
import __main__  

def clip_upper(X, *args, **kwargs):
    try:
        upper = kwargs.get("upper", None)
        if upper is not None and hasattr(X, "clip"):
            return X.clip(upper=upper)
        return X
    except Exception:
        return X

setattr(__main__, "clip_upper", clip_upper)

MODELS_DIR = Path("models")
MODEL_VERSION = "1"
MODEL_PATH = MODELS_DIR / f"model_v{MODEL_VERSION}.joblib"
META_PATH  = MODELS_DIR / f"model_v{MODEL_VERSION}.json"

_pipeline = None
_metadata = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}. "
                "Run the export notebook (05_export.ipynb) to generate it."
            )
        _pipeline = joblib.load(MODEL_PATH)
    return _pipeline

def get_metadata():
    global _metadata
    if _metadata is None:
        if META_PATH.exists():
            with open(META_PATH, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
        else:
            _metadata = {"model_version": MODEL_VERSION}
    _metadata.setdefault("model_version", MODEL_VERSION)
    return _metadata
