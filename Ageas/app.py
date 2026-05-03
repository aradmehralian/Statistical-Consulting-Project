import streamlit as st
import plotly.graph_objects as go
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
@st.cache_resource
def load_models():
    m_freq = sm.load(str(BASE_DIR / "models" / "model_freq_v2.pkl"))
    m_sev = sm.load(str(BASE_DIR / "models" / "model_sev_v3.pkl"))
    m_disc = joblib.load(str(BASE_DIR / "models" / "discretizer.pkl"))
    return m_freq, m_sev, m_disc


model_freq, model_sev, discretizer = load_models()


@st.cache_resource
def load_risk_data():
    return pd.read_csv(str(BASE_DIR / "data" / "risk_summary.csv"))


risk_summary = load_risk_data()

risk_labels = {
    0.0: "Low Risk",
    1.0: "Medium Risk",
    2.0: "High Risk",
    3.0: "Extreme Risk",
}

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
    3.0: {
        "title": "The Metropolitan Novice",
        "gender": "Male",
        "age": "~22",
        "density": "249",
        "status": "Unemployed",
        "car": "Small Car",
        "icon": "🏎️",
    },
}

risk_visuals = {
    "Low Risk": {"score": 30, "color": "#34d399"},
    "Medium Risk": {"score": 60, "color": "#fbbf24"},
    "High Risk": {"score": 90, "color": "#f87171"},
    "Extreme Risk": {"score": 100, "color": "#7f1d1d"},
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
n_years = st.sidebar.slider("Years as Customer", 0, 15, 0)

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

    visuals = risk_visuals.get(risk_tier, {"score": 0, "color": "#cccccc"})
    score = visuals["score"]
    color = visuals["color"]

    tier_stats = risk_summary[risk_summary["risk_flag"] == risk_tier]
    if not tier_stats.empty:
        tier_min = tier_stats["minimum_cost"].values[0]
        tier_max = tier_stats["maximum_cost"].values[0]
        tier_median = tier_stats["median_cost"].values[0]
    else:
        tier_min = tier_max = tier_median = 0.0

    # predicted metrics
    st.subheader("Financial Risk Breakdown")

    m1, m2, m3 = st.columns(3)
    m1.metric("Claim Frequency", f"{freq:.4f}", help="Expected claims per year")
    m2.metric("Claim Severity", f"€{sev:,.0f}", help="Average cost per claim")
    m3.metric("Expected Annual Cost", f"€{expected_cost:,.2f}")

    st.divider()

    st.markdown("### Classification")
    fig = go.Figure(
        go.Indicator(
            mode="gauge",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {
                    "range": [None, 100],
                    "visible": False,
                },
                "bar": {"color": color, "thickness": 1},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 2,
                "bordercolor": "#1e2330",
                "steps": [
                    {"range": [0, 30], "color": "rgba(52, 211, 153, 0.1)"},
                    {"range": [30, 60], "color": "rgba(251, 191, 36, 0.1)"},
                    {"range": [60, 90], "color": "rgba(248, 113, 113, 0.1)"},
                    {"range": [90, 100], "color": "rgba(127, 29, 29, 0.1)"},
                ],
            },
        )
    )

    fig.add_annotation(
        x=0.5,
        y=0.2,
        text=f"<b>{risk_tier.upper()}</b>",
        showarrow=False,
        font=dict(size=24, color=color, family="Arial Black"),
        align="center",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.container(border=True):
        st.markdown(f"#### {risk_tier} Group Benchmarks")
        st.caption("Historical performance for this specific risk segment")

        b1, b2, b3 = st.columns(3)
        b1.metric("Minimum Cost", f"€{tier_min:,.2f}")
        b2.metric("Median Cost", f"€{tier_median:,.2f}")
        b3.metric("Maximum Cost", f"€{tier_max:,.2f}")

    st.divider()

    st.markdown(f"### {profile["title"]}")

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
