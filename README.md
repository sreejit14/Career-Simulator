# Career-Simulator

A Streamlit-based interactive application for modeling and visualizing employee salary trajectories under customized career scenarios. Leverage your pre-trained salary prediction model to:

Simulate-by-year salary projections with configurable promotion timing, role changes, and education upgrades

Compare baseline vs. scenario outcomes and calculate cumulative salary lift & ROI on training investments

Visualize salary trajectories side-by-side on dynamic Plotly charts

Explore skills-gap insights with feature-importance metrics and SHAP explanations

Easily configure candidate profiles (education, experience, location, gender, age) and scenario levers (promotion year/role, education year/level, training cost)

#Project Structure
text
career_simulator/
├── app.py            # Streamlit UI: inputs, charts & layout  
├── backend.py        # Core logic: model I/O, encoding, simulation, ROI  
├── salary_model.joblib  # Pre-trained model + encoders bundle  
└── requirements.txt  # Python dependencies
#Quick Start
Clone this repository and install dependencies:

bash
git clone https://github.com/<your-username>/career_simulator.git
cd career_simulator
pip install -r requirements.txt
Add your serialized model bundle as salary_model.joblib.

#Run the app:

bash
streamlit run app.py
Features
Custom Scenarios: Set promotion timing, target role, education upgrades, and training costs

-Real-Time Visualization: Interactive Plotly line charts for salary trajectories

-ROI Analysis: Cumulative lift vs. education cost with percentage ROI

-Explainability: Global feature-importance and local SHAP insights highlight key salary drivers

-Modular Design: Separate UI (app.py) and backend (backend.py) for easy debugging and testing

#Deployment
Push to GitHub and deploy on Streamlit Community Cloud:

Commit & push all files to your repo.

In Streamlit Cloud, connect your GitHub repo, select app.py, and deploy.

Future git push updates auto-redeploy your app.
