"""
=============================================================
 DASHBOARD STREAMLIT – PREDIKSI RISIKO STUNTING BALITA
 Jalankan: streamlit run dashboard.py
=============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import joblib
import os

# ─── Konfigurasi Halaman ────────────────────────────────────
st.set_page_config(
    page_title="Deteksi Risiko Stunting Balita",
    page_icon="🧒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Kustom ─────────────────────────────────────────────
st.markdown("""
<style>
    .risk-high  { background:#ffe0e0; border-left:5px solid #e74c3c;
                  padding:16px; border-radius:8px; }
    .risk-low   { background:#e0f7ea; border-left:5px solid #2ecc71;
                  padding:16px; border-radius:8px; }
    .metric-box { background:#f0f4ff; border-radius:10px;
                  padding:14px; text-align:center; }
    h1 { color:#1a237e; }
</style>
""", unsafe_allow_html=True)

# ─── Load Model & Artefak ───────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_dir = "models"
    if not os.path.exists(model_dir):
        st.error("❌ Model belum dilatih. Jalankan `python train_model.py` terlebih dahulu.")
        st.stop()
    return {
        "rf" : joblib.load(f"{model_dir}/random_forest.pkl"),
        "lr" : joblib.load(f"{model_dir}/logistic_regression.pkl"),
        "sc" : joblib.load(f"{model_dir}/scaler.pkl"),
        "le_gender" : joblib.load(f"{model_dir}/le_gender.pkl"),
        "le_wasting": joblib.load(f"{model_dir}/le_wasting.pkl"),
        "le_target" : joblib.load(f"{model_dir}/le_target.pkl"),
    }

artifacts = load_artifacts()

# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
st.title("🧒 Sistem Deteksi Risiko Stunting Balita")
st.markdown(
    "Dashboard prediksi berbasis Machine Learning untuk mengidentifikasi "
    "risiko stunting pada anak di bawah lima tahun. "
    "Masukkan data balita pada sidebar untuk mendapatkan prediksi instan."
)
st.divider()

# ══════════════════════════════════════════════════════════
#  SIDEBAR — INPUT DATA BALITA
# ══════════════════════════════════════════════════════════
with st.sidebar:
    st.header("📋 Data Balita")
    st.caption("Masukkan data antropometri balita di bawah ini.")

    jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    umur_bulan    = st.slider("Umur (bulan)", min_value=0, max_value=60, value=24)
    tinggi_badan  = st.number_input("Tinggi Badan (cm)", min_value=30.0, max_value=130.0,
                                     value=80.0, step=0.1, format="%.1f")
    berat_badan   = st.number_input("Berat Badan (kg)", min_value=1.0, max_value=30.0,
                                     value=10.0, step=0.1, format="%.1f")
    haz_score     = st.number_input("HAZ Score (Height-for-Age Z-score)",
                                     min_value=-6.0, max_value=6.0,
                                     value=-0.5, step=0.1, format="%.2f")
    whz_score     = st.number_input("WHZ Score (Weight-for-Height Z-score)",
                                     min_value=-5.0, max_value=5.0,
                                     value=0.0, step=0.1, format="%.2f")
    status_wasting = st.selectbox("Status Wasting", ["Tidak", "Ya"])

    model_choice  = st.radio("Pilih Model Prediksi",
                              ["Random Forest (Disarankan)", "Logistic Regression"])

    predict_btn   = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

    st.divider()
    st.caption("ℹ️ HAZ < -2 → indikator klinis stunting utama")
    st.caption("ℹ️ WHZ < -2 → indikator wasting")

# ══════════════════════════════════════════════════════════
#  AREA UTAMA — TAMPILKAN HASIL
# ══════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.1, 0.9])

with col_left:
    st.subheader("📊 Ringkasan Data Masukan")

    summary = {
        "Parameter"   : ["Jenis Kelamin","Umur","Tinggi Badan","Berat Badan",
                          "HAZ Score","WHZ Score","Status Wasting"],
        "Nilai"       : [jenis_kelamin, f"{umur_bulan} bulan",
                          f"{tinggi_badan} cm", f"{berat_badan} kg",
                          f"{haz_score:.2f}", f"{whz_score:.2f}", status_wasting],
        "Interpretasi": [
            "—",
            "Usia di bawah 5 tahun" if umur_bulan <= 60 else "Di luar rentang balita",
            "Normal" if tinggi_badan >= 70 else "Perlu perhatian",
            "Normal" if berat_badan >= 7  else "Rendah",
            "Normal" if haz_score >= -2 else "⚠️ Risiko Stunting",
            "Normal" if whz_score >= -2 else "⚠️ Risiko Wasting",
            "Berisiko" if status_wasting == "Ya" else "Normal",
        ]
    }
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)

with col_right:
    st.subheader("🎯 Hasil Prediksi")

    if predict_btn:
        # ── Preprocessing input ──────────────────────────
        gender_enc  = artifacts["le_gender"].transform([jenis_kelamin])[0]
        wasting_enc = artifacts["le_wasting"].transform([status_wasting])[0]

        input_arr = np.array([[
            gender_enc, umur_bulan, tinggi_badan, berat_badan,
            haz_score,  whz_score,  wasting_enc
        ]])
        input_sc = artifacts["sc"].transform(input_arr)

        # ── Pilih model ──────────────────────────────────
        model = (artifacts["rf"]
                 if "Random Forest" in model_choice
                 else artifacts["lr"])
        model_label = ("Random Forest" if "Random Forest" in model_choice
                        else "Logistic Regression")

        pred_enc  = model.predict(input_sc)[0]
        pred_prob = model.predict_proba(input_sc)[0]
        pred_label = artifacts["le_target"].inverse_transform([pred_enc])[0]

        prob_stunting     = pred_prob[1]   # kelas "Ya"
        prob_not_stunting = pred_prob[0]   # kelas "Tidak"

        # ── Tampilkan hasil ──────────────────────────────
        if pred_label == "Ya":
            st.markdown(f"""
            <div class="risk-high">
            <h2 style="color:#c0392b; margin:0">🔴 RISIKO TINGGI STUNTING</h2>
            <p style="margin:6px 0 0 0">Balita ini <b>berpotensi mengalami stunting</b>.
            Segera konsultasikan ke tenaga kesehatan.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="risk-low">
            <h2 style="color:#27ae60; margin:0">🟢 RISIKO RENDAH</h2>
            <p style="margin:6px 0 0 0">Balita ini <b>tidak terindikasi stunting</b>.
            Pertahankan pola asuh dan gizi seimbang.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"<br>**Model digunakan:** {model_label}", unsafe_allow_html=True)

        # ── Gauge probabilitas ───────────────────────────
        fig, ax = plt.subplots(figsize=(5, 2.8))
        bar_colors = ["#2ecc71", "#e74c3c"]
        ax.barh(["Tidak Stunting","Stunting"],
                [prob_not_stunting, prob_stunting],
                color=bar_colors, height=0.45, edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probabilitas", fontsize=9)
        ax.set_title("Probabilitas Prediksi", fontsize=10, fontweight="bold")
        ax.axvline(0.5, color="grey", linestyle="--", lw=1, alpha=0.6)
        for i, v in enumerate([prob_not_stunting, prob_stunting]):
            ax.text(v + 0.02, i, f"{v*100:.1f}%", va="center", fontsize=9, fontweight="bold")
        ax.set_facecolor("#f8f9fa")
        fig.patch.set_facecolor("#f8f9fa")
        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.info("👈 Masukkan data balita di sidebar, lalu tekan **Prediksi Sekarang**.")

# ══════════════════════════════════════════════════════════
#  BAGIAN BAWAH — INFORMASI TAMBAHAN & REKOMENDASI
# ══════════════════════════════════════════════════════════
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 📌 Definisi Stunting")
    st.markdown(
        "Stunting adalah kondisi gagal tumbuh pada anak akibat kekurangan gizi "
        "kronis, ditandai tinggi badan di bawah -2 standar deviasi (SD) dari "
        "median WHO Child Growth Standards (HAZ < -2)."
    )

with col2:
    st.markdown("#### ⚠️ Faktor Risiko Utama")
    factors = [
        "HAZ Score < -2 (indikator primer)",
        "Status Wasting bersamaan",
        "Tinggi badan sangat rendah",
        "Umur < 24 bulan (1000 HPK)",
        "Berat badan lahir rendah",
    ]
    for f in factors:
        st.markdown(f"• {f}")

with col3:
    st.markdown("#### 💡 Rekomendasi Tindak Lanjut")
    recs = [
        "Pantau pertumbuhan rutin tiap bulan",
        "Konsultasi ke Posyandu/Puskesmas",
        "Pemberian ASI eksklusif (0–6 bulan)",
        "MPASI bergizi seimbang (>6 bulan)",
        "Intervensi gizi sensitif & spesifik",
    ]
    for r in recs:
        st.markdown(f"• {r}")

st.divider()
st.caption(
    "⚕️ **Disclaimer**: Dashboard ini hanya alat bantu skrining berbasis ML, "
    "bukan pengganti diagnosis medis profesional. "
    "Gunakan hasil ini sebagai referensi awal, bukan keputusan akhir. "
    "| Dataset: Simulasi berdasarkan WHO Child Growth Standards | Model: Random Forest & Logistic Regression"
)
