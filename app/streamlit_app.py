"""Interactive churn dashboard.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churn.config import load_config  # noqa: E402
from churn.data import INPUT_COLUMNS  # noqa: E402
from churn.predict import ChurnPredictor  # noqa: E402

st.set_page_config(page_title="Churn Risk Explorer", page_icon="📉", layout="wide")
cfg = load_config()


@st.cache_resource(show_spinner="Loading model...")
def get_predictor() -> ChurnPredictor:
    if not cfg.model_path.exists():
        # First launch on a fresh deploy (e.g. Streamlit Community Cloud): train a quick model.
        from churn.train import run

        with st.spinner("No trained model found. Training one now (about a minute)..."):
            run(cfg, n_trials=10, folds=3)
    return ChurnPredictor.load(cfg.model_path)


predictor = get_predictor()
m = predictor.bundle["metrics"]
econ = predictor.economics

st.title("📉 Customer Churn Risk Explorer")
st.caption(
    f"Model: **{predictor.bundle['model_name'].replace('_', ' ')}** · "
    f"test ROC-AUC **{m['test_at_chosen_threshold']['roc_auc']:.3f}** · "
    f"offer policy: **{predictor.policy.replace('_', ' ')}** · trained {predictor.bundle['trained_at']}"
)

tab_score, tab_batch, tab_model = st.tabs(["Score a customer", "Score a file", "Model performance"])

# ------------------------------------------------------------------ single customer
with tab_score:
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("Customer profile")
        c1, c2 = st.columns(2)
        tenure = c1.slider("Tenure (months)", 0, 72, 3)
        monthly = c2.number_input("Monthly charges ($)", 18.0, 120.0, 85.0, step=1.0)
        contract = c1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = c2.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        internet = c1.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        phone = c2.selectbox("Phone service", ["Yes", "No"])
        no_net = internet == "No"
        net_opt = ["No internet service"] if no_net else ["No", "Yes"]

        with st.expander("Add-on services", expanded=True):
            a1, a2, a3 = st.columns(3)
            security = a1.selectbox("Online security", net_opt)
            backup = a2.selectbox("Online backup", net_opt)
            protection = a3.selectbox("Device protection", net_opt)
            support = a1.selectbox("Tech support", net_opt)
            tv = a2.selectbox("Streaming TV", net_opt)
            movies = a3.selectbox("Streaming movies", net_opt)
            lines = a1.selectbox("Multiple lines", ["No", "Yes"] if phone == "Yes" else ["No phone service"])

        with st.expander("Demographics & billing"):
            d1, d2 = st.columns(2)
            gender = d1.selectbox("Gender", ["Female", "Male"])
            senior = d2.selectbox("Senior citizen", ["No", "Yes"])
            partner = d1.selectbox("Partner", ["No", "Yes"])
            dependents = d2.selectbox("Dependents", ["No", "Yes"])
            paperless = d1.selectbox("Paperless billing", ["Yes", "No"])

    customer = {
        "gender": gender, "SeniorCitizen": int(senior == "Yes"), "Partner": partner,
        "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
        "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
    }
    df = pd.DataFrame([customer])
    res = predictor.predict(df, explain=True, top_k=6).iloc[0]

    with right:
        st.subheader("Prediction")
        k1, k2, k3 = st.columns(3)
        k1.metric("Churn probability", f"{res.churn_probability:.0%}")
        k2.metric("Risk tier", res.risk_tier.title())
        k3.metric("Offer value", f"${res.expected_offer_value:,.0f}",
                  help="Expected $ gain from sending this customer a retention offer")
        if res.send_offer:
            st.success(f"**Send a retention offer.** Expected gain ${res.expected_offer_value:,.0f} "
                       f"after the ${econ.offer_cost:.0f} offer cost.")
        else:
            st.info("**Don't send an offer.** The expected saving doesn't cover the cost.")

        st.markdown("**Why the model thinks so**")
        reasons = pd.DataFrame(res.top_reasons)
        reasons["label"] = reasons["feature"] + " = " + reasons["value"].astype(str)
        chart = (
            alt.Chart(reasons)
            .mark_bar()
            .encode(
                x=alt.X("impact:Q", title="Impact on churn risk (SHAP, log-odds)"),
                y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(domain=["increases risk", "decreases risk"], range=["#d6604d", "#4393c3"]),
                    legend=alt.Legend(orient="bottom", title=None),
                ),
                tooltip=["feature", "value", alt.Tooltip("impact:Q", format="+.3f")],
            )
            .properties(height=240)
        )
        st.altair_chart(chart, width="stretch")

        st.markdown("**What if?** Try a retention lever and see how the risk changes.")
        w1, w2 = st.columns(2)
        new_contract = w1.selectbox("Switch contract to", ["One year", "Two year", "Month-to-month"])
        new_payment = w2.selectbox("Switch payment to", ["Credit card (automatic)", "Bank transfer (automatic)",
                                                         "Electronic check", "Mailed check"])
        whatif = df.assign(Contract=new_contract, PaymentMethod=new_payment)
        p_new = predictor.predict_proba(whatif)[0]
        st.metric("Churn probability after change", f"{p_new:.0%}",
                  delta=f"{(p_new - res.churn_probability) * 100:+.1f} pts", delta_color="inverse")

# ------------------------------------------------------------------ batch file
with tab_batch:
    st.subheader("Score a customer list")
    st.write("Upload a CSV with the same columns as the Telco dataset. You'll get back a prioritized "
             "call list, sorted by the expected value of contacting each customer.")
    file = st.file_uploader("CSV file", type="csv")
    if file is not None:
        raw = pd.read_csv(file)
        missing = [c for c in INPUT_COLUMNS if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
        else:
            from churn.data import clean

            data = clean(raw)
            scored = pd.concat([data.reset_index(drop=True), predictor.predict(data)], axis=1)
            scored = scored.sort_values("expected_offer_value", ascending=False)
            n_offer = int(scored["send_offer"].sum())
            b1, b2, b3 = st.columns(3)
            b1.metric("Customers scored", f"{len(scored):,}")
            b2.metric("Recommended for an offer", f"{n_offer:,}")
            b3.metric("Expected campaign value",
                      f"${scored.loc[scored.send_offer, 'expected_offer_value'].sum():,.0f}")
            st.dataframe(scored, width="stretch", height=400)
            st.download_button("Download scored CSV", scored.to_csv(index=False), "scored_customers.csv")

# ------------------------------------------------------------------ model card
with tab_model:
    st.subheader("How good is the model?")
    t = m["test_at_chosen_threshold"]
    c = st.columns(4)
    c[0].metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    c[1].metric("PR-AUC", f"{t['pr_auc']:.3f}", help=f"Random guessing scores {1 - m['test_majority_class_accuracy']:.2f}")
    c[2].metric("Churners in top 20% riskiest", f"{m['top_20pct_churners_caught']:.0%}")
    c[3].metric("Profit per 1,000 customers", f"${m['deployed_policy_profit_per_1000']:,.0f}",
                delta=f"${m['deployed_policy_profit_per_1000'] - m['default_threshold_profit_per_1000']:,.0f} vs 0.5 threshold")

    reports = cfg.reports_dir
    if (reports / "policy_comparison.csv").exists():
        st.markdown("**Targeting policies on the held-out test set**")
        st.dataframe(pd.read_csv(reports / "policy_comparison.csv"), width="stretch", hide_index=True)
    if (reports / "model_comparison.csv").exists():
        st.markdown("**Model comparison (cross-validated)**")
        st.dataframe(pd.read_csv(reports / "model_comparison.csv"), width="stretch", hide_index=True)

    figs = reports / "figures"
    names = ["roc_pr", "profit_curve", "gains", "shap_importance", "calibration"]
    cols = st.columns(2)
    for i, name in enumerate(n for n in names if (figs / f"{n}.png").exists()):
        cols[i % 2].image(str(figs / f"{name}.png"))

    with st.expander("Business assumptions"):
        st.json(json.loads(json.dumps(predictor.bundle["economics"])))
        st.caption("Edit these in config.yaml and retrain to see how the targeting policy changes.")
