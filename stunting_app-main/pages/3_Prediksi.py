import streamlit as st
import pandas as pd
import joblib

st.title("🔍 Prediksi Risiko Stunting")

model = joblib.load(
    "model/model.pkl"
)

target_encoder = joblib.load(
    "model/target_encoder.pkl"
)

gender_encoder = joblib.load(
    "model/gender_encoder.pkl"
)

col1, col2 = st.columns(2)

with col1:

    gender = st.selectbox(
        "Jenis Kelamin",
        ["M", "F"]
    )

    age = st.number_input(
        "Umur (bulan)",
        min_value=1,
        max_value=60,
        value=24
    )

with col2:

    weight = st.number_input(
        "Berat Badan (kg)",
        min_value=1.0,
        max_value=40.0,
        value=10.0
    )

    height = st.number_input(
        "Tinggi Badan (cm)",
        min_value=30.0,
        max_value=150.0,
        value=80.0
    )

st.divider()

if st.button(
    "Prediksi",
    use_container_width=True
):

    gender_value = gender_encoder.transform(
        [gender]
    )[0]

    data = pd.DataFrame({
        "Gender": [gender_value],
        "Age (Month)": [age],
        "Weight": [weight],
        "Height": [height]
    })

    prediction = model.predict(data)

    probability = model.predict_proba(
        data
    )

    result = target_encoder.inverse_transform(
        prediction
    )[0]

    confidence = max(
        probability[0]
    ) * 100

    st.subheader(
        "📋 Hasil Prediksi"
    )

    if result == "Stunted":

        st.error(
            f"⚠️ Risiko Stunting ({confidence:.2f}%)"
        )

    else:

        st.success(
            f"✅ Tidak Stunting ({confidence:.2f}%)"
        )

    st.progress(
        confidence / 100
    )

    st.write(
        f"Tingkat Keyakinan Model: {confidence:.2f}%"
    )

    st.divider()

    if result == "Stunted":

        st.warning("""
Rekomendasi:

- Konsultasi dengan tenaga kesehatan
- Monitoring pertumbuhan rutin
- Perbaikan asupan gizi
""")

    else:

        st.info("""
Pertumbuhan balita terindikasi normal.

Tetap lakukan pemantauan berkala.
""")