# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import backend as be  # local import

# ---------- Sidebar inputs ----------
st.sidebar.header("🎛️ Candidate profile")
profile = {
    "Education": st.sidebar.selectbox("Highest education", be.EDU_ORDER, 1),
    "Experience": st.sidebar.slider("Years of experience", 0, 30, 5),
    "Location": st.sidebar.radio("Location", ["Urban", "Suburban", "Rural"], 0),
    "Job_Title": st.sidebar.selectbox("Current role", be.JOB_PATHS, 0),
    "Age": st.sidebar.slider("Current age", 20, 64, 25),
    "Gender": st.sidebar.radio("Gender", ["Male", "Female"], horizontal=True),
}
base_row = pd.Series(profile)

st.sidebar.markdown("---")
st.sidebar.header("🛣️ Scenario planning")
years = st.sidebar.slider("Years to simulate", 1, 20, 10)
promo_year = st.sidebar.slider("Promotion year", 0, years, 5)
promo_role = st.sidebar.selectbox("Role after promotion", be.JOB_PATHS, 2)
edu_year = st.sidebar.slider("Education upgrade year", 0, years, 3)
edu_target = st.sidebar.selectbox("Target education", be.EDU_ORDER, 2)
edu_cost = st.sidebar.number_input("Cost of education ($)", 0, 200000, 20000, 1000)

# ---------- Run simulation ----------
baseline_df = be.simulate(
    base_row, years,
    promo_year=years + 1, promo_title=profile["Job_Title"],
    edu_year=years + 1, edu_target=profile["Education"]
)
scenario_df = be.simulate(
    base_row, years,
    promo_year, promo_role, edu_year, edu_target
)
lift, roi = be.salary_lift_and_roi(baseline_df, scenario_df, edu_cost)

# ---------- Main page ----------
st.title("🚀 Career Simulator")

col1, col2 = st.columns(2)
col1.metric("Current salary", f"${be.predict_salary(base_row):,.0f}")
col2.metric(
    f"Salary after {years} yr",
    f"${scenario_df.iloc[-1]['Predicted_Salary']:,.0f}",
    f"{scenario_df.iloc[-1]['Predicted_Salary'] - baseline_df.iloc[-1]['Predicted_Salary']:,.0f}"
)

fig = px.line(
    scenario_df, x="Year", y="Predicted_Salary", title="Salary trajectory (scenario)"
)
fig.add_scatter(
    x=baseline_df["Year"],
    y=baseline_df["Predicted_Salary"],
    name="Baseline",
    line=dict(dash="dash", color="gray")
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("💰 ROI analysis")
st.write(f"Total salary lift: **${lift:,.0f}**")  # years span already obvious
st.write(f"Education cost: **${edu_cost:,.0f}**")
st.write(f"ROI: **{roi:.1f}%**")
st.caption("Powered by Streamlit • Plotly • scikit-learn")
