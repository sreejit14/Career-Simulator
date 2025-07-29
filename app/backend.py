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
    df = df.copy()
    
    # Experience groups
    df['Experience_Group'] = pd.cut(
        df['Experience'],
        bins=[0, 5, 15, float('inf')],
        labels=['Early_Career', 'Mid_Career', 'Late_Career'],
        include_lowest=True)
    
    # Binary flags for career stages
    df['Early_Career'] = (df['Experience_Group'] == 'Early_Career').astype(int)
    df['Mid_Career'] = (df['Experience_Group'] == 'Mid_Career').astype(int)
    df['Late_Career'] = (df['Experience_Group'] == 'Late_Career').astype(int)
    
    # Polynomial and ratio features
    df['Experience_Sq'] = df['Experience'] ** 2
    df['Experience_Per_Age'] = df['Experience'] / (df['Age'] + 1e-6)  # Add small epsilon to prevent division by zero
    
    # Manager/Director flag
    df['Manager_Director'] = df['Job_Title'].isin(['Manager', 'Director']).astype(int)
    
    # Education-Experience combinations
    df['HighExp_LowEdu'] = ((df['Experience'] > 15) & (df['Education'] == 'High School')).astype(int)
    df['LowExp_HighEdu'] = ((df['Experience'] < 5) & (df['Education'].isin(['PhD', 'Master']))).astype(int)
    
    # Interaction features
    df['Education_Job_Interaction'] = df['Education'] + '_' + df['Job_Title']
    df['PhD_Experience'] = ((df['Education'] == 'PhD') & (df['Experience'] > 10)).astype(int)
    
    return df



def predict_salary(row: pd.Series) -> float:
    model = load_model()
    X = pd.DataFrame([row])
    X = add_engineered_features(X)
    
    # Debug: Print the data being sent to the model
    print("Columns:", X.columns.tolist())
    print("Data types:", X.dtypes.to_dict())
    print("Sample values:", X.iloc[0].to_dict())
    print("Any NaN values:", X.isnull().sum().sum())
    print("Any infinite values:", np.isinf(X.select_dtypes(include=[np.number])).sum().sum())
    
    try:
        prediction = model.predict(X)
        return float(prediction[0])
    except Exception as e:
        print(f"Model prediction error: {e}")
        print(f"Error type: {type(e)}")
        raise e

# ---------- Simulator ----------
def simulate(
    base_row: pd.Series,
    years: int,
    promo_year: int,
    promo_title: str,
    edu_year: int,
    edu_target: str,
) -> pd.DataFrame:
    # Only use the original 6 features for the base row
    # The engineered features will be added by add_engineered_features() in predict_salary()
    features = ["Education", "Experience", "Location", "Job_Title", "Age", "Gender"]
    
    records = []
    for y in range(years + 1):
        r = base_row.copy()
        r["Year"] = y
        r["Experience"] = base_row["Experience"] + y
        r["Age"] = base_row["Age"] + y
        
        # Apply promotions and education changes
        if y >= promo_year:
            r["Job_Title"] = promo_title
        if y >= edu_year:
            r["Education"] = edu_target
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
