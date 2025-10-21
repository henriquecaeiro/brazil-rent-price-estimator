# Brazil Rent Price Estimator

## 📖 Summary

This project is a rent price estimator for properties in Brazil, specifically focused on the city of São Paulo. It uses a machine learning model to predict rent prices based on property features and location. The model is served through a FastAPI application.

## 🎯 Goal

The main goal of this project is to provide a reliable and accurate way to estimate rent prices in São Paulo. This is a personal project to showcase machine learning skills, from data exploration and feature engineering to model training and deployment. This can be useful for tenants, landlords, and real estate professionals to make informed decisions.

## ⚙️ How to Execute

To run this project, you need to have Python 3.11 or higher installed.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/brazil-rent-price-estimator.git
   cd brazil-rent-price-estimator
   ```
2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate # On Windows, use `.venv\Scripts\activate`
   ```
3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the API:**
   ```bash
   uvicorn api.main:app --reload
   ```
The API will be available at `http://127.0.0.1:8000`.

## 🤖 Model and Metrics

The rent price prediction is done by a `StackingRegressor` model, which combines the predictions of a `RandomForestRegressor` and a `CatBoostRegressor`. The final prediction is made by a `Ridge` regressor. The model also uses a `KMeans` model to create geographic clusters based on latitude and longitude.

The main metrics used to evaluate the model are:

| Metric | Value |
| :--- | :--- |
| **Mean Absolute Error (MAE)** | R$ 998.47 |
| **Median Absolute Error (MedAE)** | R$ 588.84 |
| **R-squared (R2)** | 0.7793 |

## 📊 Dataset

The dataset used in this project is not updated and is from **May 1st, 2023**. It is used for **educational purposes only**.

The dataset was obtained from Kaggle: [São Paulo Housing Prices](https://www.kaggle.com/datasets/renatosn/sao-paulo-housing-prices).

## Endpoints

The API has the following endpoints:

*   `GET /`: Returns the status of the API.
*   `POST /predict`: Predicts the rent price based on the input data.
*   `GET /unique/{column}`: Returns the unique values for a given column (`address`, `district`, or `type`).

### Example of a `POST /predict` request body:

```json
{
    "address": "Avenida Paulista",
    "district": "Bela Vista",
    "area": 100,
    "bedrooms": 2,
    "garage": 1,
    "type": "Apartamento",
    "city": "São Paulo"
}
```
