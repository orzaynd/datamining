import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load Dataset
df = pd.read_excel("data/Overall Data.xlsx")

st.title("📊 Dashboard Dataset Stunting")
st.markdown("Ringkasan data yang digunakan untuk pelatihan model Machine Learning.")

# KPI
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Total Data",
        value=f"{len(df):,}"
    )

with col2:
    st.metric(
        label="Jumlah Gender",
        value=df["Gender"].nunique()
    )

with col3:
    st.metric(
        label="Usia Minimum",
        value=f"{int(df['Age (Month)'].min())} Bulan"
    )

with col4:
    st.metric(
        label="Usia Maksimum",
        value=f"{int(df['Age (Month)'].max())} Bulan"
    )

st.divider()

# Grafik
col_left, col_right = st.columns(2)

with col_left:

    st.subheader("Distribusi Status Stunting")

    fig1, ax1 = plt.subplots(figsize=(5, 4))

    df["Height for Age"].value_counts().plot(
        kind="bar",
        ax=ax1
    )

    st.pyplot(fig1)

with col_right:

    st.subheader("Distribusi Gender")

    fig2, ax2 = plt.subplots(figsize=(5, 4))

    df["Gender"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        ax=ax2
    )

    ax2.set_ylabel("")

    st.pyplot(fig2)

st.divider()

st.subheader("Distribusi Umur Balita")

fig3, ax3 = plt.subplots(figsize=(8, 4))

ax3.hist(
    df["Age (Month)"],
    bins=20
)

ax3.set_xlabel("Umur (Bulan)")
ax3.set_ylabel("Jumlah Data")

st.pyplot(fig3)

st.divider()

st.subheader("Preview Dataset")

st.dataframe(
    df.head(10),
    use_container_width=True
)