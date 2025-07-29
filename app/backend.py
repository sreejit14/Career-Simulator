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
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # Make a copy to avoid side effects
    df = df.copy()
    
    # Experience groups - example bins, adjust as needed
    df['Experience_Group'] = pd.cut(
        df['Experience'],
        bins=[0, 5, 15, 30, 100],
        labels=['Early_Career', 'Mid_Career', 'Late_Career', 'Expert'],
        include_lowest=True,
        right=False
    )

    df['Early_Career'] = (df['Experience_Group'] == 'Early_Career').astype(int)
    df['Mid_Career'] = (df['Experience_Group'] == 'Mid_Career').astype(int)
    df['Late_Career'] = (df['Experience_Group'] == 'Late_Career').astype(int)

    # Derived features
    df['Experience_Sq'] = df['Experience'] ** 2
    df['Experience_Per_Age'] = df['Experience'] / (df['Age'] + 1e-4)  # avoid division by zero

    # Manager or Director flag
    df['Manager_Director'] = df['Job_Title'].isin(['Manager', 'Director']).astype(int)

    # Combined logic flags
    df['HighExp_LowEdu'] = ((df['Experience'] > 15) & (df['Education'] == 'High School')).astype(int)
    df['LowExp_HighEdu'] = ((df['Experience'] < 5) & (df['Education'].isin(['PhD', 'Master']))).astype(int)
    df['Education_Job_Interaction'] = df['Education'] + '_' + df['Job_Title']
    df['PhD_Experience'] = ((df['Education'] == 'PhD') & (df['Experience'] > 10)).astype(int)

    return df

def predict_salary(row: pd.Series) -> float:
    model = load_model()
    X = pd.DataFrame([row])
    X = add_engineered_features(X) 
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
