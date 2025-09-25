# schemas_fixed.py
from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator

PropertyType = Literal["Apartamento", "Casa", "Casa em condomínio", "Studio e kitnet"]

class RentItem(BaseModel):
    area: Optional[int] = Field(None, ge=1, le=10000)
    bedrooms: Optional[int] = Field(None, ge=1, le=10)
    garage: Optional[int] = Field(None, ge=0, le=10)
    address: Optional[str] = Field(None, max_length=128)
    district: Optional[str] = Field(None, max_length=128)
    type: Optional[PropertyType] = None

    @field_validator("address", "district", mode="before")
    @classmethod
    def normalize_text(cls, v):
        if isinstance(v, str):
            v = v.strip()
            return v.lower()
        return v

    model_config = {"protected_namespaces": ()}

class PredictRequest(BaseModel):
    items: Optional[List[RentItem]] = None
    model_config = {"protected_namespaces": ()}

class PredictResponseItem(BaseModel):
    prediction: float
    warning: Optional[str] = None
    model_config = {"protected_namespaces": ()}

class PredictResponse(BaseModel):
    model_version: str
    predictions: List[PredictResponseItem]
    inference_ms: float | None = None
    model_config = {"protected_namespaces": ()}
