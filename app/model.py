# app/model.py

import lightgbm as lgb
import polars as pl
import numpy as np
from dateutil import parser
import os
from pathlib import Path

# --- UPDATED PATH: Model is now at the project root ---
# Path(__file__).parent.parent gets the project root directory from /app/model.py
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "lightgbm_power_model_2.txt"

model = lgb.Booster(model_file=MODEL_PATH)

def transform_datetime_features(pl_df: pl.DataFrame):
    dt_series = pl_df['Datetime']
    dt_objects = [parser.parse(dt) for dt in dt_series]
    months = [dt.month for dt in dt_objects]
    minutes = [dt.hour * 60 + dt.minute for dt in dt_objects]
    month_sin = [np.sin(2 * np.pi * m / 12) for m in months]
    month_cos = [np.cos(2 * np.pi * m / 12) for m in months]
    time_sin = [np.sin(2 * np.pi * t / 1440) for t in minutes]
    time_cos = [np.cos(2 * np.pi * t / 1440) for t in minutes]

    pl_df_transformed = pl_df.with_columns([
        pl.Series('Month_sin', month_sin),
        pl.Series('Month_cos', month_cos),
        pl.Series('Time_sin', time_sin),
        pl.Series('Time_cos', time_cos)
    ])
    return pl_df_transformed, dt_objects

def run_prediction_for_date(date: str):
    # --- UPDATED PATH: Test data is now at the project root ---
    file_path = PROJECT_ROOT / "test" / f"{date}.csv"

    if not file_path.exists():
        print(f"Error: Data file not found for date {date} at {file_path}.")
        return None

    try:
        df_original = pl.read_csv(file_path)
        features_df, dt_objects = transform_datetime_features(df_original.clone())
        X_predict = features_df.select([
            'Month_sin', 'Month_cos', 'Time_sin', 'Time_cos',
            'Temperature', 'Humidity', 'WindSpeed',
            'GeneralDiffuseFlows', 'DiffuseFlows'
        ]).to_pandas()

        predictions = model.predict(X_predict)
        results = {
            "timestamps": [dt.strftime("%H:%M") for dt in dt_objects],
            "predicted_values": predictions.tolist(),
            "actual_values": df_original['PowerConsumption'].to_list()
        }
        return results
    except Exception as e:
        print(f"Error processing forecast for date {date}: {e}")
        return None