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

## 💾 Generating the Model

The machine learning model is not saved in this repository. To use the application, you first need to generate the model and auxiliary files by running the Jupyter Notebooks in the correct order. The `notebooks/features.py` script is used by the notebooks to create new features.

**Important:** The first time you run the `model_export.ipynb` notebook, it will perform a geocoding step to get the latitude and longitude for the addresses. This process can be slow and will create a `rents_geocoded.csv` file in the `data/work/` directory. Subsequent runs will use this file and be much faster.

1.  **Open the `notebooks` directory.**
2.  **Run the `eda.ipynb` notebook:** This notebook performs exploratory data analysis and saves the cleaned data in the `data/work/` directory.
3.  **Run the `model_export.ipynb` notebook:** This notebook trains the final model and saves it as `stacking_model.joblib` in the `models/` directory. It also saves a `kmeans_model.joblib` file, which is used for geo-clustering.

After running these notebooks, the following files will be created:
- `data/work/rents_geocoded.csv`
- `models/stacking_model.joblib`
- `models/kmeans_model.joblib`

The API will then be able to use these files for predictions.

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
