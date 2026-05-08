import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import statsmodels.api as sm
import joblib
from pathlib import Path
import numpy as np
import shap
from IPython.display import display, Javascript
import threading

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
def get_dtype_template():
    return pd.DataFrame(
        {
            "gender": pd.Series(dtype="object"),
            "carType": pd.Series(dtype="object"),
            "job": pd.Series(dtype="object"),
            "cover": pd.Series(dtype="int64"),
            "nYears": pd.Series(dtype="float64"),
            "age": pd.Series(dtype="float64"),
            "density": pd.Series(dtype="float64"),
            "carVal": pd.Series(dtype="float64"),
        }
    )


@st.cache_resource
def load_risk_data():
    return pd.read_csv(str(BASE_DIR / "data" / "risk_summary.csv"))


@st.cache_resource
def load_profile_data():
    return pd.read_csv(str(BASE_DIR / "data" / "tier_profile.csv"), index_col=0)


@st.cache_resource
def get_background_data():
    df_train = pd.read_csv(str(BASE_DIR / "data" / "combined_data.csv"))
    cols = ["gender", "carType", "cover", "job", "nYears", "age", "density", "carVal"]

    return df_train[cols].sample(200, random_state=25)


def generate_cost_shap(model_freq, model_sev, df_input):
    feature_cols = [
        "gender",
        "carType",
        "cover",
        "job",
        "nYears",
        "age",
        "density",
        "carVal",
    ]

    background = get_background_data().copy()

    background["gender"] = background["gender"].astype(str)
    background["carType"] = background["carType"].astype(str)
    background["job"] = background["job"].astype(str)

    def predict_cost(X):
        template = get_dtype_template()

        X_df = pd.DataFrame(X, columns=feature_cols)

        for col in template.columns:
            X_df[col] = X_df[col].astype(template[col].dtype)

        freq = model_freq.predict(X_df)
        sev = model_sev.predict(X_df)

        return (freq * sev).values

    explainer = shap.KernelExplainer(predict_cost, background.values)
    shap_values = explainer.shap_values(df_input.values, nsamples=100)

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=explainer.expected_value,
        data=df_input.values[0],
        feature_names=feature_cols,
    )

    return explanation


def plotly_shap_like_waterfall(explanation, title=""):
    values = explanation.values
    base = explanation.base_values
    features = explanation.feature_names
    data = explanation.data

    order = np.argsort(np.abs(values))[::-1]
    values = values[order]
    features = [features[i] for i in order]
    data = [data[i] for i in order]

    # Build labels
    variable_labels = {
        "age": "Age",
        "cover": "Cover",
        "nYears": "Years as Customer",
        "carVal": "Car Value",
        "density": "Density",
        "carType": "Car Type",
        "job": "Job",
        "gender": "Gender",
    }

    labels = [
        (
            f"{variable_labels[f]} = {"Yes" if v == 1 else "No"}"
            if f == "cover"
            else f"{variable_labels[f]} = {v}"
        )
        for f, v in zip(features, data)
    ]

    # Cumulative positions
    x_start = base
    xs = []
    widths = []
    colors = []
    text = []

    for val in values:
        xs.append(x_start)
        widths.append(val)
        colors.append("#FF0051" if val >= 0 else "#1E88E5")
        text.append(f"{val:+.2f}")
        x_start += val

    final_value = x_start

    fig = go.Figure()

    for i in range(len(values)):
        fig.add_trace(
            go.Bar(
                x=[widths[i]],
                y=[labels[i]],
                base=xs[i],
                orientation="h",
                marker=dict(color=colors[i]),
                text=[text[i]],
                textposition="auto",
                insidetextanchor="middle",
                hovertemplate=f"{labels[i]}<br>{text[i]}<extra></extra>",
                showlegend=False,
            )
        )

    fig.add_vline(
        x=base,
        line=dict(color="gray", dash="dash"),
        annotation_text=f"Global Average = {base:.2f}",
        annotation_position="bottom left",
        annotation_xshift=-5,
    )

    fig.add_vline(
        x=final_value,
        line=dict(color="gray", dash="dot"),
        annotation_text=f"Expected Cost = {final_value:.2f}",
        annotation_position="top right",
        annotation_yanchor="bottom",
    )

    fig.update_layout(
        title=title,
        barmode="overlay",
        height=400 + len(values) * 30,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_title="Expected Cost (€)",
        yaxis=dict(autorange="reversed"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=None),
        template="plotly_dark" if _is_dark_mode() else "plotly_white",
    )

    return fig


def _is_dark_mode():
    try:
        result = {}
        done = threading.Event()

        def _callback(data):
            result["dark"] = data
            done.set()

        js = Javascript("""
            const bg = getComputedStyle(document.body).backgroundColor;
            const m = bg.match(/\d+/g);
            const lum = m ? (0.299*m[0] + 0.587*m[1] + 0.114*m[2]) / 255 : 1;
            IPython.notebook.kernel.execute(
                `_dark_mode_result = ${lum < 0.5}`
            );
        """)
        display(js)
        return False
    except Exception:
        return False


risk_summary = load_risk_data()
tier_profile = load_profile_data()


risk_labels = {
    0.0: "Low Risk",
    1.0: "Medium Risk",
    2.0: "High Risk",
    3.0: "Extreme Risk",
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
age = st.sidebar.slider("Age", 18, 80, 30)
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
car_val = st.sidebar.number_input(
    "Car Value (€)", min_value=500, max_value=50_000, value=10_000, step=1000
)
n_years = st.sidebar.slider("Years as Customer", 0, 15, 0)
cover = int(st.sidebar.toggle("Cover", help="e.g. fire, theft, etc."))

run = st.sidebar.button("Score Customer")


# main panel
st.title("Customer Profile")

if run:
    df_input = pd.DataFrame(
        [
            {
                "gender": gender,
                "carType": car_type,
                "cover": cover,
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
    m2.metric("Claim Severity", f"€{sev:,.0f}", help="Expected cost per claim")
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

    with st.expander("SHAP Explanation (Expected Cost)", expanded=False):
        with st.spinner("Computing SHAP values..."):
            try:
                exp_cost = generate_cost_shap(model_freq, model_sev, df_input)

                fig = plotly_shap_like_waterfall(exp_cost)
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Cost SHAP failed: {e}")

    with st.container(border=True):
        st.markdown(f"#### {risk_tier} Group Benchmarks")
        st.caption("Historical performance for this specific risk segment")

        b1, b2, b3 = st.columns(3)
        b1.metric("Minimum Cost", f"€{tier_min:,.2f}")
        b2.metric("Median Cost", f"€{tier_median:,.2f}")
        b3.metric("Maximum Cost", f"€{tier_max:,.2f}")


else:
    st.info(
        "\U0001f448 Fill in the customer profile on the left and click **Score Customer**."
    )
