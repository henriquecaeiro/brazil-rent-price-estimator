# -*- coding: utf-8 -*-
"""Pydantic schemas for the API."""

from typing import Optional, Literal, List
from pydantic import BaseModel, Field, field_validator

# Define the allowed property types.
PropertyType = Literal["Apartment", "House", "Condominium House", "Studio and kitnet"]

class RentItem(BaseModel):
    """Pydantic model for a single rent item."""
    area: Optional[int] = Field(None, ge=1, le=10000)
    bedrooms: Optional[int] = Field(None, ge=1, le=10)
    garage: Optional[int] = Field(None, ge=0, le=10)
    address: Optional[str] = Field(None, max_length=128)
    district: Optional[str] = Field(None, max_length=128)
    type: Optional[PropertyType] = None

    @field_validator("address", "district", mode="before")
    @classmethod
    def normalize_text(cls, v):
        """Normalize text fields by stripping whitespace and converting to lowercase."""
        if isinstance(v, str):
            v = v.strip()
            return v.lower()
        return v

    model_config = {"protected_namespaces": ()}

class PredictRequest(BaseModel):
    """Pydantic model for a prediction request."""
    items: Optional[List[RentItem]] = None
    model_config = {"protected_namespaces": ()}

class PredictResponseItem(BaseModel):
    """Pydantic model for a single prediction response item."""
    prediction: float
    warning: Optional[str] = None
    model_config = {"protected_namespaces": ()}

class PredictResponse(BaseModel):
    """Pydantic model for a prediction response."""
    model_version: str
    predictions: List[PredictResponseItem]
    inference_ms: float | None = None
    model_config = {"protected_namespaces": ()}
    

class UniqueValuesResponse(BaseModel):
    """Pydantic model for the response of the unique values endpoint."""
    column: str
    values: List[str]
