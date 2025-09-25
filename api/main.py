from fastapi import FastAPI, HTTPException
import logging
import time
import pandas as pd
from typing import Optional

from .schemas import PredictRequest, PredictResponse, PredictResponseItem
from . import loader

logger = logging.getLogger("rent-api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Rent Price Estimator API", version="1.0.0")

@app.get("/health")
def health():
    meta = loader.get_metadata()
    return {"status": "ok", "model_version": meta.get("model_version")}

@app.get("/model-info")
def model_info():
    return loader.get_metadata()

@app.post("/predict", response_model=PredictResponse)
def predict(req: Optional[PredictRequest] = None):
    if (req is None) or (not req.items):
        raise HTTPException(status_code=400, detail="Empty payload: 'items' is required.")

    df = pd.DataFrame([i.model_dump() for i in req.items])

    meta = loader.get_metadata()
    clip_p99 = (meta.get("preprocess") or {}).get("clip_area_p99")
    
    if "area" in df.columns:
        df["area"] = pd.to_numeric(df["area"], errors="coerce")

    if clip_p99 is not None and "area" in df.columns:
        lim = float(clip_p99)
        too_big = df["area"].notna() & (df["area"] > lim)
        if too_big.any():
            idxs = df.index[too_big].tolist()
            raise HTTPException(
                status_code=422,
                detail=(f"area exceeds training p99 ({lim}) at rows {idxs}. "
                        "Please send realistic values.")
            )

    t0 = time.time()
    try:
        y_pred = loader.get_pipeline().predict(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e!s}")
    infer_ms = (time.time() - t0) * 1000.0

    predictions = [PredictResponseItem(prediction=float(v)) for v in y_pred]
    logger.info("predict batch=%d inference_ms=%.2f", len(predictions), infer_ms)

    return PredictResponse(
        model_version=str(meta.get("model_version", "")),
        predictions=predictions,
        inference_ms=round(infer_ms, 2),
    )
