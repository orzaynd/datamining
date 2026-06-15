import streamlit as st

st.title("ℹ️ Tentang Dataset")

st.markdown("""

### Informasi Dataset

| Keterangan            | Nilai          |
| --------------------- | -------------- |
| Data Awal             | 40.071         |
| Data Setelah Cleaning | 19.908         |
| Target                | Height for Age |
| Algoritma             | Random Forest  |

### Variabel Input

* Gender
* Age (Month)
* Weight
* Height

### Target

* Stunted
* Not Stunted

### Hasil Model

* Accuracy : 96.79%
* Precision : 96.79%
* Recall : 98.41%
* F1 Score : 97.59%
* ROC AUC : 99.54%
* Cross Validation : 95.19%
  """)
