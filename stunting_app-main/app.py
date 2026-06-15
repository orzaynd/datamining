import streamlit as st

st.set_page_config(
page_title="Stunting Predictor",
page_icon="👶",
layout="wide"
)

st.title("👶 Sistem Prediksi Risiko Stunting Balita")

st.markdown("""
Aplikasi berbasis Machine Learning menggunakan algoritma Random Forest untuk membantu mendeteksi risiko stunting pada balita berdasarkan data antropometri.
""")

col1,col2,col3 = st.columns(3)

col1.metric(
"Accuracy",
"96.79%"
)

col2.metric(
"ROC AUC",
"99.54%"
)

col3.metric(
"Cross Validation",
"95.19%"
)

st.divider()

st.info("""
Gunakan menu di sidebar untuk:

* Dashboard
* Evaluasi Model
* Prediksi Stunting
* Tentang Dataset
  """)
