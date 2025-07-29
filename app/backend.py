# backend.py

import joblib
from pathlib import Path
import numpy as np
import pandas as pd

# ---------- Constants ----------
EDU_ORDER = ["High School", "Bachelor", "Master", "PhD"]
JOB_PATHS = ["Analyst", "Engineer", "Manager", "Director"]

# ---------- Model I/O ----------
_model_bundle = None

def load_bundle(path: str = "salary_predictor_final.joblib"):
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"{path} not found. Place your joblib bundle here.")
    _model_bundle = joblib.load(path)
    return _model_bundle

def _encode_row(bundle: dict, row: pd.Series) -> np.ndarray:
    le_edu = bundle["le_education"]
    le_loc = bundle["le_location"]
    le_job = bundle["le_job_title"]
    le_gen = bundle["le_gender"]


    return np.array([
        le_edu.transform([row["Education"]])[0],
        row["Experience"],
        le_loc.transform([row["Location"]])[0],
        le_job.transform([row["Job_Title"]])[0],
        row["Age"],
        le_gen.transform([row["Gender"]])[0],
    ]).reshape(1, -1)

def predict_salary(bundle: dict, row: pd.Series) -> float:
    """
    Predict salary for a single profile row.
    """
    X = _encode_row(bundle, row)
    return float(bundle["model"].predict(X)[0])

# ---------- Simulator ----------
def simulate(
    bundle: dict,
    base_row: pd.Series,
    years: int,
    promo_year: int,
    promo_title: str,
    edu_year: int,
    edu_target: str,
) -> pd.DataFrame:
    """
    Build a DataFrame with yearly salary projections under the given scenario.
    """
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

        r["Predicted_Salary"] = predict_salary(bundle, r)
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
