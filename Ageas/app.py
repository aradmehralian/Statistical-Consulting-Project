import streamlit as st
import pandas as pd
import statsmodels.api as sm
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Insurance Customer Profiling",
    page_icon=str(BASE_DIR / "images" / "insurance.png"),
    layout="wide",
    initial_sidebar_state="auto",
)

# the models created in the model.ipynb file
model_freq = sm.load(str(BASE_DIR / "models" / "model_freq_v2.pkl"))
model_sev = sm.load(str(BASE_DIR / "models" / "model_sev_v3.pkl"))
discretizer = joblib.load(str(BASE_DIR / "models" / "discretizer.pkl"))


risk_labels = {0.0: "Low Risk", 1.0: "Medium Risk", 2.0: "High Risk"}

profiles = {
    0.0: {
        "title": "The Rural Veteran",
        "gender": "Male",
        "age": "~46",
        "density": "69",
        "status": "Employed",
        "car": "Large Car",
        "icon": "🚜",
    },
    1.0: {
        "title": "The Suburban Professional",
        "gender": "Male",
        "age": "~29",
        "density": "140",
        "status": "Employed",
        "car": "Medium Car",
        "icon": "🏢",
    },
    2.0: {
        "title": "The Young Urbanite",
        "gender": "Male",
        "age": "~23",
        "density": "225",
        "status": "Unemployed",
        "car": "Small Car",
        "icon": "🌆",
    },
}


# side bar for entering customer info
st.sidebar.header("Customer Info")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 99, 30)
job = st.sidebar.selectbox(
    "Employment", ["Employed", "Unemployed", "Self-employed", "Retired", "Housewife"]
)
density = st.sidebar.number_input(
    "Population Density (inh/km²)",
    min_value=0,
    max_value=300,
    value=140,
    step=10,
)
car_type = st.sidebar.selectbox("Car Type", ["A", "B", "C", "D", "E"])
car_cat = st.sidebar.selectbox("Car Category", ["Small", "Medium", "Large"])
car_val = st.sidebar.number_input(
    "Car Value (€)", min_value=500, max_value=50_000, value=10_000, step=1000
)
n_years = st.sidebar.slider("Years as Customer", 0, 10, 0)

run = st.sidebar.button("Score Customer")


# main panel
st.title("Customer Profile")

if run:
    df_input = pd.DataFrame(
        [
            {
                "gender": gender,
                "carType": car_type,
                "carCat": car_cat,
                "job": job,
                "nYears": n_years,
                "age": age,
                "density": density,
                "carVal": car_val,
            }
        ]
    )

    freq = model_freq.predict(df_input).iloc[0]
    sev = model_sev.predict(df_input).iloc[0]
    expected_cost = freq * sev

    risk_bin = discretizer.transform(
        pd.DataFrame([[expected_cost]], columns=["Expected_Cost"])
    )[0][0]
    risk_tier = risk_labels[risk_bin]
    profile = profiles[risk_bin]

    # predicted metrics
    st.subheader("Predicted Cost Breakdown")
    col1, col2, col3 = st.columns(3)
    col1.metric("Claim Rate (claims/year)", f"{freq:.4f}")
    col2.metric("Severity per Claim (€)", f"€{sev:,.0f}")
    col3.metric("Expected Cost (€)", f"€{expected_cost:,.0f}")

    st.divider()
    st.markdown("### Classification")
    if "High" in risk_tier:
        st.error(f"{risk_tier}")
    elif "Medium" in risk_tier:
        st.warning(f"{risk_tier}")
    else:
        st.success(f"{risk_tier}")
    st.divider()

    st.markdown(f"### {profile["icon"]} {profile["title"]}")

    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            st.caption("DEMOGRAPHICS")
            st.write(f"**Age:** {profile["age"]} years")
            st.write(f"**Gender:** {profile["gender"]}")
            st.write(f"**Status:** {profile["status"]}")

        with col2:
            st.caption("ENVIRONMENT & ASSETS")
            st.write(f"**Density:** {profile['density']} inh/km²")
            st.write(f"**Vehicle:** {profile["car"]}")

        st.info("This profile represents the statistical average for this risk tier.")

else:
    st.info(
        "\U0001f448 Fill in the customer profile on the left and click **Score Customer**."
    )
