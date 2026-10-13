"""Interactive Streamlit predictions and exact SHAP explanations."""

from pathlib import Path
import hashlib
import logging
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

st.set_page_config(page_title="Machine Failure Explorer", page_icon="⚙️", layout="wide")

MODEL_PATH = Path(__file__).resolve().parent / "predictive_maintenance.joblib"
RAW_COLUMNS = [
    "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
    "Torque [Nm]", "Tool wear [min]",
]
EXPECTED_COLUMNS = RAW_COLUMNS + ["Power [W]", "Temperature difference"]
SHORT_NAMES = [
    "Air temperature", "Process temperature", "Rotational speed",
    "Torque", "Tool wear", "Power", "Temperature difference",
]
UNITS = ["K", "K", "rpm", "Nm", "min", "W", "K"]
TRAINING_RANGES = [(295.3, 304.5), (305.7, 313.8), (1168, 2861), (4.6, 76.2), (0, 253)]
INPUT_KEYS = ["air", "process", "speed", "torque", "wear"]
EXAMPLES = {
    "Lower-score example": [298.1, 308.6, 1551, 42.8, 0],
    "Failure-alert example": [302.6, 310.9, 1331, 52.0, 154],
}
RED, BLUE, NAVY = "#D94A55", "#287DC0", "#344E73"
CHART_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "toImageButtonOptions": {"format": "png", "scale": 2},
}


@st.cache_resource
def load_bundle(stamp):
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    if list(bundle["feature_columns"]) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected saved feature columns.")
    if list(model.feature_names_in_) != EXPECTED_COLUMNS or set(model.classes_) != {0, 1}:
        raise ValueError("Unexpected model schema.")
    if not np.isfinite(bundle["cutoff"]) or not 0 <= bundle["cutoff"] <= 1:
        raise ValueError("Invalid saved cutoff.")
    background = bundle["shap_background"]
    if list(background.columns) != EXPECTED_COLUMNS or len(background) == 0:
        raise ValueError("The saved SHAP reference data are missing or incompatible.")
    if not np.isfinite(background.to_numpy()).all():
        raise ValueError("The reference data contain invalid values.")
    return bundle


def build_features(values, columns=EXPECTED_COLUMNS):
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != 5 or not np.isfinite(array).all():
        raise ValueError("Enter five finite readings per record.")
    readings = pd.DataFrame(array, columns=RAW_COLUMNS)
    readings["Power [W]"] = (
        readings["Torque [Nm]"] * readings["Rotational speed [rpm]"] * (2 * np.pi / 60)
    )
    readings["Temperature difference"] = (
        readings["Process temperature [K]"] - readings["Air temperature [K]"]
    )
    return readings[columns]


def failure_scores(features, bundle):
    index = list(bundle["model"].classes_).index(1)
    return bundle["model"].predict_proba(features)[:, index]


@st.cache_resource
def load_explainer(stamp):
    bundle = load_bundle(stamp)
    def predict_failure(masked_values):
        features = pd.DataFrame(masked_values, columns=EXPECTED_COLUMNS)
        return failure_scores(features, bundle)

    # Seven features permit exact enumeration of the 128 feature coalitions.
    # Strict equality avoids treating close readings on opposite sides of a
    # tree split as interchangeable during the masker's invariant optimization.
    class StrictIndependentMasker(shap.maskers.Independent):
        def invariants(self, x):
            return np.equal(self.data, x)

    return shap.ExactExplainer(
        predict_failure,
        StrictIndependentMasker(bundle["shap_background"], max_samples=len(bundle["shap_background"])),
    )


def explain_frame(features, stamp):
    bundle = load_bundle(stamp)
    explanation = load_explainer(stamp)(features, max_evals=128, silent=True)
    contributions = np.asarray(explanation.values, dtype=float)
    baselines = np.asarray(explanation.base_values, dtype=float)
    scores = failure_scores(features, bundle)
    if contributions.shape != features.shape:
        raise ValueError("Unexpected SHAP output shape.")
    if not np.isfinite(contributions).all() or not np.isfinite(baselines).all():
        raise ValueError("SHAP returned invalid values.")
    if not np.allclose(baselines + contributions.sum(axis=1), scores, atol=1e-6, rtol=0):
        raise ValueError("SHAP contributions did not reconstruct the model output.")
    return scores, baselines, contributions


@st.cache_data(max_entries=256, show_spinner=False)
def explain_values(values, stamp):
    features = build_features(values)
    scores, bases, contributions = explain_frame(features, stamp)
    return {
        "values": list(values), "features": features,
        "score": float(scores[0]), "baseline": float(bases[0]),
        "contributions": contributions[0],
    }


@st.cache_data(max_entries=4, show_spinner=False)
def explain_reference(stamp):
    background = load_bundle(stamp)["shap_background"].copy()
    scores, bases, contributions = explain_frame(background, stamp)
    return background, scores, bases, contributions


def chart_style(fig, height=440):
    fig.update_layout(
        template="plotly_white", height=height,
        margin=dict(l=30, r=35, t=55, b=45),
        font=dict(size=13), hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def render_chart(fig, key):
    st.plotly_chart(
        fig, key=key, width="stretch", config=CHART_CONFIG,
    )


def score_gauge(score, cutoff):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"valueformat": ".3f"},
        title={"text": "Failure model score"},
        gauge={
            "axis": {"range": [0, 1], "tickvals": [0, cutoff, 0.5, 1]},
            "bar": {"color": RED if score >= cutoff else BLUE},
            "steps": [
                {"range": [0, cutoff], "color": "#E9F2FB"},
                {"range": [cutoff, 1], "color": "#FBECEF"},
            ],
            "threshold": {"value": cutoff, "line": {"color": "#142438", "width": 4}},
        },
    ))
    return chart_style(fig, 270)


def contribution_table(result):
    return pd.DataFrame({
        "Feature": EXPECTED_COLUMNS,
        "Input value": result["features"].iloc[0].to_numpy(),
        "SHAP contribution": result["contributions"],
    })


def waterfall_chart(result, cutoff):
    order = np.argsort(-np.abs(result["contributions"]), kind="stable")
    base, score = result["baseline"], result["score"]
    contributions = result["contributions"][order]
    values = result["features"].iloc[0].to_numpy()
    names = ["Reference score"] + [
        f"{SHORT_NAMES[i]} = {values[i]:,.2f} {UNITS[i]}" for i in order
    ] + ["Final model score"]
    cumulative = base + np.cumsum(contributions)
    details = [f"Reference score: {base:.6f}"] + [
        f"{EXPECTED_COLUMNS[i]}: {values[i]:,.4f}<br>"
        f"SHAP contribution: {result['contributions'][i]:+.6f}<br>"
        f"Running score: {cumulative[j]:.6f}"
        for j, i in enumerate(order)
    ] + [f"Final model score: {score:.6f}"]
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * 7 + ["total"],
        x=[base] + contributions.tolist() + [0],
        y=names, customdata=details,
        text=[f"{base:.3f}"] + [f"{v:+.3f}" for v in contributions] + [f"{score:.3f}"],
        textposition="outside", hovertemplate="%{customdata}<extra></extra>",
        increasing={"marker": {"color": RED}},
        decreasing={"marker": {"color": BLUE}},
        totals={"marker": {"color": NAVY}},
        connector={"line": {"color": "#B8C2CF"}},
    ))
    fig.update_yaxes(autorange="reversed")
    low = min(0.0, base, float(cumulative.min()))
    high = max(1.0, base, float(cumulative.max()))
    fig.update_xaxes(title="Model score", range=[low - 0.08, high + 0.10])
    fig.add_vline(x=cutoff, line_dash="dot", line_color="#64748B",
                  annotation_text=f"Alert cutoff {cutoff:.2f}", annotation_position="top")
    fig.update_layout(title="How the readings build this prediction", showlegend=False)
    return chart_style(fig, 520)


def contributions_chart(result):
    order = np.argsort(-np.abs(result["contributions"]), kind="stable")
    contributions = result["contributions"][order]
    values = result["features"].iloc[0].to_numpy()[order]
    fig = go.Figure(go.Bar(
        orientation="h", x=contributions.tolist(),
        y=[SHORT_NAMES[i] for i in order],
        marker_color=[RED if v >= 0 else BLUE for v in contributions],
        customdata=[[float(value), UNITS[i]] for value, i in zip(values, order)],
        text=[f"{v:+.4f}" for v in contributions], textposition="outside",
        hovertemplate="%{y}<br>Reading: %{customdata[0]:,.3f} %{customdata[1]}"
                      "<br>SHAP contribution: %{x:+.6f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    magnitude = max(float(np.abs(contributions).max()) * 1.25, 0.01)
    fig.update_xaxes(title="Change in model score", range=[-magnitude, magnitude])
    fig.add_vline(x=0, line_color="#475569", line_width=1)
    fig.update_layout(title="Which readings raise or lower the score?")
    return chart_style(fig, 460)


def render_explanation(result, cutoff, prefix):
    chart_type = st.radio(
        "Explanation chart", ["Waterfall", "Contribution bars"],
        horizontal=True, key=f"{prefix}_chart",
    )
    fig = waterfall_chart(result, cutoff) if chart_type == "Waterfall" else contributions_chart(result)
    render_chart(fig, f"{prefix}_shap_plot")
    st.caption("Red raises the score; blue lowers it. Hover for exact values, drag to zoom, and double-click to reset.")
    impact = float(np.sum(result["contributions"]))
    st.write(
        f"**Reference score {result['baseline']:.4f}** + "
        f"**net contribution {impact:+.4f}** = **model score {result['score']:.4f}**."
    )
    largest = int(np.argmax(np.abs(result["contributions"])))
    direction = "raises" if result["contributions"][largest] >= 0 else "lowers"
    st.write(
        f"The largest contribution here is **{SHORT_NAMES[largest]}**. "
        f"It {direction} the score by **{abs(result['contributions'][largest]):.4f}**, "
        "relative to the reference examples."
    )
    with st.expander("Exact readings and contributions"):
        table = contribution_table(result)
        st.dataframe(table, hide_index=True, width="stretch")
        export = table.copy()
        export["Reference score"] = result["baseline"]
        export["Final model score"] = result["score"]
        export["Alert cutoff"] = cutoff
        st.download_button(
            "Download this explanation (CSV)", export.to_csv(index=False).encode("utf-8"),
            file_name="prediction_shap_explanation.csv", mime="text/csv", key=f"{prefix}_download",
        )
    st.caption(
        "The reference is the average model score over the saved training background. "
        "SHAP describes the model's reasoning; related inputs can share attribution."
    )


def render_scenario(result, bundle, stamp):
    st.subheader("What changes if one reading changes?")
    sensor_index = st.selectbox(
        "Reading to vary", list(range(5)), index=2,
        format_func=lambda i: RAW_COLUMNS[i], key="scenario_sensor",
    )
    original_value = result["values"][sensor_index]
    low, high = TRAINING_RANGES[sensor_index]
    low, high = min(low, original_value), max(high, original_value)
    signature = hashlib.sha256(repr(tuple(result["values"])).encode()).hexdigest()[:12]
    slider_key = f"scenario_{signature}_{sensor_index}"
    if sensor_index in (2, 4):
        new_value = st.slider(
            RAW_COLUMNS[sensor_index], min_value=int(low), max_value=int(high),
            value=int(original_value), step=1, key=slider_key,
        )
    else:
        new_value = st.slider(
            RAW_COLUMNS[sensor_index], min_value=float(low), max_value=float(high),
            value=float(original_value), step=0.1, key=slider_key,
        )
    changed = list(result["values"])
    changed[sensor_index] = new_value
    with st.spinner("Updating the scenario and its SHAP explanation…"):
        scenario = explain_values(tuple(changed), stamp)
    first, second = st.columns(2)
    first.metric("Submitted score", f"{result['score']:.3f}")
    second.metric("Scenario score", f"{scenario['score']:.3f}",
                  delta=f"{scenario['score'] - result['score']:+.3f}", delta_color="inverse")
    cutoff = float(bundle["cutoff"])
    if scenario["score"] >= cutoff:
        st.warning("Scenario: Failure alert")
    else:
        st.success("Scenario: No failure alert")
    grid = np.linspace(low, high, 61)
    if sensor_index in (2, 4):
        grid = np.unique(np.rint(grid).astype(int))
    variants = np.tile(np.asarray(result["values"], dtype=float), (len(grid), 1))
    variants[:, sensor_index] = grid
    curve_scores = failure_scores(build_features(variants), bundle)
    fig = go.Figure(go.Scatter(
        x=grid.tolist(), y=curve_scores.tolist(), mode="lines",
        name="Model response", line={"color": BLUE, "width": 3, "shape": "hv"},
        hovertemplate="Reading: %{x:,.2f}<br>Model score: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[original_value, new_value], y=[result["score"], scenario["score"]],
        mode="markers", name="Compared readings",
        text=["Submitted reading", "Scenario reading"],
        marker={"size": [10, 14], "symbol": ["circle", "diamond"], "color": [NAVY, RED]},
        hovertemplate="%{text}<br>Reading: %{x:,.2f}<br>Score: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=cutoff, line_dash="dot", annotation_text=f"Alert cutoff {cutoff:.2f}")
    fig.update_layout(xaxis_title=RAW_COLUMNS[sensor_index], yaxis_title="Model score",
                      title="Sampled model response as this reading changes", legend={"orientation": "h"})
    fig.update_yaxes(range=[-0.03, 1.03])
    render_chart(chart_style(fig, 350), "scenario_curve")
    st.caption(
        "Other raw readings stay fixed. Power and temperature difference are recalculated. "
        "The curve samples up to 61 values; it is a model simulation, not evidence of a causal effect."
    )
    st.subheader("SHAP explanation of the scenario")
    render_explanation(scenario, cutoff, "scenario")


def render_reference(stamp):
    with st.spinner("Explaining the saved reference examples…"):
        background, scores, bases, contributions = explain_reference(stamp)
    st.subheader(f"Feature importance across {len(background)} reference examples")
    st.caption(
        "This view summarizes the saved training reference sample. "
        "It is not the validation sample used in your notebook's global SHAP chart."
    )
    importance = np.abs(contributions).mean(axis=0)
    order = np.argsort(-importance, kind="stable")
    fig = go.Figure(go.Bar(
        orientation="h", y=[SHORT_NAMES[i] for i in order], x=importance[order].tolist(),
        marker_color=BLUE,
        hovertemplate="%{y}<br>Mean absolute SHAP contribution: %{x:.6f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Average absolute contribution to the model score",
                      title="Average size of each feature's influence")
    render_chart(chart_style(fig, 410), "reference_importance")
    index = st.selectbox(
        "Inspect a feature", list(range(7)),
        format_func=lambda i: EXPECTED_COLUMNS[i], key="reference_feature",
    )
    fig = go.Figure(go.Scatter(
        x=background.iloc[:, index].tolist(), y=contributions[:, index].tolist(),
        mode="markers", customdata=scores.tolist(),
        marker={
            "size": 9, "opacity": 0.85, "color": scores.tolist(),
            "colorscale": [[0, BLUE], [1, RED]], "cmin": 0, "cmax": 1,
            "colorbar": {"title": "Model score"},
        },
        hovertemplate="Reading: %{x:,.3f}<br>SHAP contribution: %{y:+.6f}"
                      "<br>Model score: %{customdata:.4f}<extra></extra>",
    ))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(
        title=f"{SHORT_NAMES[index]} in the reference examples",
        xaxis_title=EXPECTED_COLUMNS[index], yaxis_title="SHAP contribution to failure score",
    )
    render_chart(chart_style(fig, 390), "reference_dependence")
    st.caption("Each dot is one reference record. The bar chart uses absolute contributions; the scatter plot retains their direction.")


def choose_example(name):
    for key, value in zip(INPUT_KEYS, EXAMPLES[name]):
        st.session_state[key] = value
    st.session_state.pop("prediction_values", None)


st.title("Machine Failure Explorer")
st.caption("Predictive maintenance · Interactive predictions · SHAP explanations")
st.write("Enter five readings, select **Predict and explain**, then explore how the model reached its result.")

try:
    model_stat = MODEL_PATH.stat()
    stamp = (model_stat.st_mtime_ns, model_stat.st_size)
    bundle = load_bundle(stamp)
except FileNotFoundError:
    st.error("Place predictive_maintenance.joblib beside app.py, then restart the app.")
    st.stop()
except InconsistentVersionWarning:
    st.error("Use Python 3.12 and the supplied requirements.txt so the model's scikit-learn version matches.")
    st.stop()
except Exception:
    logging.exception("Model bundle could not be loaded")
    st.error("The model bundle could not be loaded. Check the supplied model file and requirements.txt.")
    st.stop()

for key, value in zip(INPUT_KEYS, EXAMPLES["Lower-score example"]):
    st.session_state.setdefault(key, value)

left, right = st.columns(2)
with left:
    st.button("Load lower-score example", on_click=choose_example, args=("Lower-score example",), width="stretch")
with right:
    st.button("Load failure-alert example", on_click=choose_example, args=("Failure-alert example",), width="stretch")

with st.form("sensor_readings"):
    st.subheader("Sensor readings")
    first, second, third = st.columns(3)
    with first:
        air = st.number_input("Air temperature (K)", min_value=200.0, max_value=500.0, step=0.1, format="%.1f", key="air",
                              help="Kelvin = Celsius + 273.15.")
        process = st.number_input("Process temperature (K)", min_value=200.0, max_value=600.0, step=0.1, format="%.1f", key="process")
    with second:
        speed = st.number_input("Rotational speed (rpm)", min_value=1, max_value=10000, step=1, key="speed")
        torque = st.number_input("Torque (Nm)", min_value=0.0, max_value=300.0, step=0.1, format="%.1f", key="torque")
    with third:
        wear = st.number_input("Tool wear (minutes)", min_value=0, max_value=1000, step=1, key="wear")
        st.caption("The app calculates power and temperature difference automatically.")
    submitted = st.form_submit_button("Predict and explain", type="primary", width="stretch")

if submitted:
    st.session_state.prediction_values = (air, process, speed, torque, wear)

if "prediction_values" in st.session_state:
    try:
        with st.spinner("Calculating the prediction and SHAP contributions…"):
            result = explain_values(st.session_state.prediction_values, stamp)
        score, cutoff = result["score"], float(bundle["cutoff"])
        st.divider()
        st.subheader("Last submitted prediction")
        outside = [
            name for name, value, (low, high) in zip(RAW_COLUMNS, result["values"], TRAINING_RANGES)
            if not low <= value <= high
        ]
        if outside:
            st.warning("Outside the ranges seen during training: " + ", ".join(outside) + ". This result is an extrapolation.")
        left, right = st.columns([1, 2])
        with left:
            render_chart(score_gauge(score, cutoff), "score_gauge")
        with right:
            st.metric("Model score", f"{score:.3f}")
            if score >= cutoff:
                st.warning(f"Failure alert — the score meets or exceeds the {cutoff:.2f} cutoff.")
            else:
                st.success(f"No failure alert — the score is below the {cutoff:.2f} cutoff.")
            st.caption("This 0–1 model score has not been calibrated as a real-world failure probability.")
            st.caption("The black mark on the gauge is the saved alert cutoff. Changing a scenario never changes this cutoff.")
        view = st.radio(
            "Explore the result", ["SHAP explanation", "What-if simulator", "Reference importance"],
            horizontal=True, key="result_view",
        )
        if view == "SHAP explanation":
            render_explanation(result, cutoff, "prediction")
        elif view == "What-if simulator":
            render_scenario(result, bundle, stamp)
        else:
            render_reference(stamp)
    except Exception:
        logging.exception("Prediction or explanation failed")
        st.error("The prediction or explanation could not be completed. Confirm Python 3.12 and all supplied requirements, then check the deployment logs.")
else:
    st.info("Load an example or enter readings, then select Predict and explain.")

with st.expander("About the project and test results"):
    st.write(
        "The saved Random Forest uses five sensor readings plus calculated power and temperature difference. "
        "The 0.3 cutoff was selected using validation F1. SHAP uses the saved 100 training reference examples."
    )
    st.table(pd.DataFrame({
        "Test metric": ["Precision", "Recall", "F1", "Average precision (AP)"],
        "Result": ["79.7%", "75.0%", "77.3%", "0.828"],
    }))
    st.write("On 2,000 test records: 51 failures detected, 17 missed, and 13 false alarms. The test set was also checked during the initial baseline experiment.")
    st.caption("Academic prototype using simulated AI4I data. It classifies recorded conditions; advance warning time and real factory performance have not been established.")

