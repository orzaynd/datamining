import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt

metrics = joblib.load(
    "model/metrics.pkl"
)

st.title("📈 Evaluasi Model")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Accuracy",
    f"{metrics['accuracy']*100:.2f}%"
)

col2.metric(
    "Precision",
    f"{metrics['precision']*100:.2f}%"
)

col3.metric(
    "Recall",
    f"{metrics['recall']*100:.2f}%"
)

col4, col5, col6 = st.columns(3)

col4.metric(
    "F1 Score",
    f"{metrics['f1']*100:.2f}%"
)

col5.metric(
    "ROC AUC",
    f"{metrics['roc_auc']*100:.2f}%"
)

col6.metric(
    "Cross Validation",
    f"{metrics['cv_mean']*100:.2f}%"
)

st.divider()

st.subheader("Confusion Matrix")

cm = metrics["cm"]

cm_df = pd.DataFrame(
    cm,
    columns=[
        "Pred Not Stunted",
        "Pred Stunted"
    ],
    index=[
        "Actual Not Stunted",
        "Actual Stunted"
    ]
)

st.dataframe(
    cm_df,
    use_container_width=True
)

if metrics["feature_importance"] is not None:

    st.subheader(
        "Feature Importance"
    )

    features = [
        "Gender",
        "Age",
        "Weight",
        "Height"
    ]

    fig, ax = plt.subplots()

    ax.barh(
        features,
        metrics["feature_importance"]
    )

    st.pyplot(fig)