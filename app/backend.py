# backend.py

import joblib
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Constants ----------
EDU_ORDER = ["High School", "Bachelor", "Master", "PhD"]
JOB_PATHS = ["Analyst", "Engineer", "Manager", "Director"]

# ---------- Model I/O ----------
_model = None

def load_model(path: str = "app/salary_predictor_final.pkl"):
    global _model
    if _model is not None:
        return _model
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"{path} not found. Place your model pipeline here.")
    _model = joblib.load(path)
    return _model

def predict_salary(row: pd.Series) -> float:
    """
    Predict salary for a single profile row.
    """
    model = load_model()
    X = pd.DataFrame([row])
    return float(model.predict(X)[0])

# ---------- Simulator ----------
def simulate(
    base_row: pd.Series,
    years: int,
    promo_year: int,
    promo_title: str,
    edu_year: int,
    edu_target: str,
) -> pd.DataFrame:
    features = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender"]
    records = []
    for y in range(years + 1):
        r = base_row.copy()
        r["Year"] = y
        r["Experience"] = base_row["Experience"] + y
        r["Age"] = base_row["Age"] + y
        if y >= promo_year:
            r["Job_Title"] = promo_title
        if y >= edu_year:
            r["Education"] = edu_target
        # Use only the actual features for prediction!
        row_for_pred = r[features]
        r["Predicted_Salary"] = predict_salary(row_for_pred)
        records.append(r)
    return pd.DataFrame(records)


def salary_lift_and_roi(
    baseline_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    edu_cost: float
) -> tuple[float, float]:
    """
    Compute cumulative salary lift and ROI (%) for education/training.
    """
    total_lift = scenario_df["Predicted_Salary"].sum() - baseline_df["Predicted_Salary"].sum()
    if edu_cost <= 0:
        return total_lift, float("nan")
    roi_pct = (total_lift - edu_cost) / edu_cost * 100
    return total_lift, roi_pct
