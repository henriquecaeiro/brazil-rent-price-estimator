# -*- coding: utf-8 -*-
"""Main module of the FastAPI API for rent prediction."""

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Literal
from .schemas import UniqueValuesResponse

from .loader import load_model
from . import loader

from geopy.geocoders import Nominatim
import ssl
import certifi

# --- Initial Configuration ---
# Create a FastAPI instance with a title, description, and version.
app = FastAPI(
    title="Rent Prediction API",
    description="API to estimate the rental value of properties in São Paulo.",
    version="1.1.0" # Updated version
)

# --- Model Loading ---
# Load the trained models from the models directory.
try:
    stacking_model = load_model("stacking_model.joblib")
    kmeans_model = load_model("kmeans_model.joblib")
except FileNotFoundError as e:
    print(e)
    stacking_model = None
    kmeans_model = None

# --- Geocoder and Cache Configuration ---
# Configure the geocoder with a custom user agent and SSL context.
ctx = ssl.create_default_context(cafile=certifi.where())
geolocator = Nominatim(user_agent="api_rent_sp_robust", ssl_context=ctx)

# Dictionary to store geocoding results and avoid repeated calls.
geocode_cache = {}

# --- Input Data Model (Pydantic) ---
# Define the input data model for the rent prediction endpoint.
class RentInput(BaseModel):
    address: str
    district: str
    area: int
    bedrooms: int
    garage: int
    type: str
    city: str = "São Paulo"

# --- API Endpoints ---
@app.get("/")
def read_root():
    """Root endpoint to check if the API is online."""
    return {"status": "API online"}

@app.post("/predict")
def predict_rent(rent_input: RentInput):
    """Predict the rent price based on the input data."""
    # Check if the models were loaded successfully.
    if not stacking_model or not kmeans_model:
        return {"error": "Models were not loaded. Prediction cannot be performed."}

    # --- Geocoding with Fallback Logic ---
    # Try to geocode the full address first.
    full_address = f"{rent_input.address}, {rent_input.district}, {rent_input.city}, Brazil"
    location = None
    
    # Check the cache first.
    if full_address in geocode_cache:
        location = geocode_cache[full_address]
        print(f"Address found in cache: {full_address}")
    else:
        try:
            location = geolocator.geocode(full_address, timeout=10)
            geocode_cache[full_address] = location # Save to cache, even if None
        except Exception as e:
            print(f"Geocoding API failed for full address: {e}")
            
    # Fallback Logic: if the full address fails, try with just the district.
    if not location:
        print(f"Failed to find full address. Trying fallback to district: {rent_input.district}")
        fallback_address = f"{rent_input.district}, {rent_input.city}, Brazil"
        
        if fallback_address in geocode_cache:
            location = geocode_cache[fallback_address]
            print(f"District found in cache: {fallback_address}")
        else:
            try:
                location = geolocator.geocode(fallback_address, timeout=10)
                geocode_cache[fallback_address] = location # Save to cache
            except Exception as e:
                 print(f"Geocoding API failed for district: {e}")

    # If even the fallback fails, return an error.
    if not location:
        return {"error": "Address not found, even with fallback to district. Prediction could not be performed."}

    lat, lon = location.latitude, location.longitude
    
    # --- Prediction Pipeline ---
    # Create a dataframe with the coordinates to predict the geo cluster.
    coords_df = pd.DataFrame([[lat, lon]], columns=['latitude', 'longitude'])
    geo_cluster = kmeans_model.predict(coords_df)[0]

    # Create a dataframe with the input data for the stacking model.
    input_data = {
        'address': [rent_input.address], 'district': [rent_input.district],
        'type': [rent_input.type], 'area': [rent_input.area],
        'bedrooms': [rent_input.bedrooms], 'garage': [rent_input.garage],
        'latitude': [lat], 'longitude': [lon], 'geo_cluster': [geo_cluster]
    }
    input_df = pd.DataFrame(input_data)
    
    # Predict the rent price using the stacking model.
    try:
        prediction = stacking_model.predict(input_df)
        predicted_value = round(prediction[0], 2)
        
        return {
            "value": predicted_value,
        }
    except Exception as e:
        return {"error": f"Error during prediction: {e}"}
    
@app.get("/unique/{column}", response_model=UniqueValuesResponse, tags=["unique"])
def unique_values(column: Literal["address", "district", "type"]):
    """Get the unique values for a given column in the dataset."""
    try:
        values = loader.get_unique_values(column)
        return UniqueValuesResponse(column=column, values=values)
    except FileNotFoundError as e:
        # Handle the case where the CSV file is not found.
        raise HTTPException(status_code=500, detail=str(e))
    except (ValueError, KeyError) as e:
        # Handle the case where the column is not found.
        raise HTTPException(status_code=400, detail=str(e))
    except pd.errors.EmptyDataError:
        # Handle the case where the CSV file is empty.
        raise HTTPException(status_code=500, detail="CSV is empty or unreadable.")
    except Exception as e:
        # Handle any other exceptions.
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
