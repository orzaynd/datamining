"""
=============================================================
 SISTEM KLASIFIKASI RISIKO STUNTING BALITA
 Menggunakan dataset: dataset_stunting_balita.xlsx
 Model: Logistic Regression & Random Forest Classifier
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_auc_score
)

# ─── Konfigurasi ───────────────────────────────────────────
DATASET_PATH = "project/dataset_stunting_balita.xlsx"
OUTPUT_DIR   = "project/output"
MODEL_DIR    = "project/models"
RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ══════════════════════════════════════════════════════════
# BAGIAN 1 — DATA LOADING & EXPLORATORY DATA ANALYSIS (EDA)
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("  BAGIAN 1: DATA LOADING & EDA")
print("=" * 60)

df = pd.read_excel(DATASET_PATH, sheet_name="Dataset Stunting")
print(f"\n✓ Dataset dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
print("\n── Sampel data (5 baris pertama) ──")
print(df.head())

print("\n── Informasi tipe data & nilai kosong ──")
print(df.info())

print("\n── Statistik deskriptif ──")
print(df.describe())

print("\n── Distribusi label target (Stunting) ──")
vc = df["Stunting"].value_counts()
print(vc)

# ── Visualisasi 1: Distribusi Label ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Exploratory Data Analysis – Stunting Balita", fontsize=14, fontweight="bold")

colors = ["#2ecc71", "#e74c3c"]
axes[0].bar(vc.index, vc.values, color=colors, edgecolor="white", linewidth=1.2)
axes[0].set_title("Distribusi Kasus Stunting")
axes[0].set_xlabel("Status Stunting")
axes[0].set_ylabel("Jumlah Balita")
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 3, str(v), ha="center", fontweight="bold")

# ── Visualisasi 2: HAZ Score vs Status Stunting ─────────────
for status, color in zip(["Tidak", "Ya"], colors):
    subset = df[df["Stunting"] == status]["HAZ (Height-for-Age Z-score)"]
    axes[1].hist(subset, bins=25, alpha=0.7, label=status, color=color, edgecolor="white")
axes[1].axvline(-2, color="black", linestyle="--", lw=1.5, label="Batas HAZ = -2")
axes[1].set_title("Distribusi HAZ Score")
axes[1].set_xlabel("HAZ (Height-for-Age Z-score)")
axes[1].set_ylabel("Frekuensi")
axes[1].legend()

# ── Visualisasi 3: Boxplot Tinggi Badan vs Status ───────────
df.boxplot(column="Tinggi_Badan_cm", by="Stunting", ax=axes[2],
           patch_artist=True,
           boxprops=dict(facecolor="#3498db", color="navy"),
           medianprops=dict(color="red", lw=2))
axes[2].set_title("Tinggi Badan vs Status Stunting")
axes[2].set_xlabel("Status Stunting")
axes[2].set_ylabel("Tinggi Badan (cm)")
plt.sca(axes[2]); plt.title("Tinggi Badan vs Stunting")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_eda_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Plot EDA disimpan → {OUTPUT_DIR}/01_eda_overview.png")

# ══════════════════════════════════════════════════════════
# BAGIAN 2 — FEATURE ENGINEERING & PREPROCESSING
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BAGIAN 2: FEATURE ENGINEERING & PREPROCESSING")
print("=" * 60)

# Pilih fitur yang relevan secara klinis
FEATURE_COLS = [
    "Jenis_Kelamin",
    "Umur_Bulan",
    "Tinggi_Badan_cm",
    "Berat_Badan_kg",
    "HAZ (Height-for-Age Z-score)",
    "WHZ (Weight-for-Height Z-score)",
    "Wasting",
]
TARGET_COL = "Stunting"

# Encode variabel kategorikal
le_gender  = LabelEncoder()
le_wasting = LabelEncoder()
le_target  = LabelEncoder()

df["Jenis_Kelamin_enc"] = le_gender.fit_transform(df["Jenis_Kelamin"])   # L=0 P=1
df["Wasting_enc"]       = le_wasting.fit_transform(df["Wasting"])        # Tidak=0 Ya=1
df["Stunting_enc"]      = le_target.fit_transform(df["Stunting"])        # Tidak=0 Ya=1

print(f"  Gender mapping  : {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
print(f"  Wasting mapping : {dict(zip(le_wasting.classes_, le_wasting.transform(le_wasting.classes_)))}")
print(f"  Target mapping  : {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Fitur numerik untuk model
NUMERIC_FEATURES = [
    "Jenis_Kelamin_enc",
    "Umur_Bulan",
    "Tinggi_Badan_cm",
    "Berat_Badan_kg",
    "HAZ (Height-for-Age Z-score)",
    "WHZ (Weight-for-Height Z-score)",
    "Wasting_enc",
]

FEATURE_NAMES = [
    "Jenis Kelamin",
    "Umur (Bulan)",
    "Tinggi Badan (cm)",
    "Berat Badan (kg)",
    "HAZ Score",
    "WHZ Score",
    "Status Wasting",
]

X = df[NUMERIC_FEATURES].values
y = df["Stunting_enc"].values

# Heatmap korelasi
plt.figure(figsize=(9, 7))
corr_df = df[NUMERIC_FEATURES + ["Stunting_enc"]].rename(
    columns=dict(zip(NUMERIC_FEATURES + ["Stunting_enc"],
                     FEATURE_NAMES + ["Stunting"]))
)
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
            mask=mask, linewidths=0.5, vmin=-1, vmax=1)
plt.title("Heatmap Korelasi Fitur – Stunting Balita", fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Heatmap korelasi disimpan → {OUTPUT_DIR}/02_correlation_heatmap.png")

# Train-test split (stratified karena imbalanced)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\n✓ Train: {X_train.shape[0]} sampel | Test: {X_test.shape[0]} sampel")

# Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("✓ Scaling selesai (StandardScaler)")

# ══════════════════════════════════════════════════════════
# BAGIAN 3 — PEMODELAN
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BAGIAN 3: PEMODELAN")
print("=" * 60)

# ── Model A: Logistic Regression ────────────────────────────
lr_model = LogisticRegression(
    max_iter=500,
    class_weight="balanced",   # atasi imbalanced
    random_state=RANDOM_STATE,
    solver="lbfgs"
)
lr_model.fit(X_train_sc, y_train)
print("✓ Logistic Regression dilatih")

# ── Model B: Random Forest ───────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_model.fit(X_train_sc, y_train)
print("✓ Random Forest dilatih")

# ══════════════════════════════════════════════════════════
# BAGIAN 4 — EVALUASI & PERBANDINGAN MODEL
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BAGIAN 4: EVALUASI")
print("=" * 60)

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    """Hitung metrik evaluasi lengkap untuk satu model."""
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]

    cv_scores = cross_val_score(model, X_tr, y_tr, cv=StratifiedKFold(5), scoring="f1")

    metrics = {
        "Model"    : name,
        "Accuracy" : accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall"   : recall_score(y_te, y_pred, zero_division=0),
        "F1-Score" : f1_score(y_te, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_te, y_prob),
        "CV F1 (mean)": cv_scores.mean(),
    }

    print(f"\n── {name} ──")
    for k, v in metrics.items():
        if k != "Model":
            print(f"   {k:<18}: {v:.4f}")
    print(f"\n{classification_report(y_te, y_pred, target_names=['Tidak Stunting','Stunting'])}")

    return metrics, confusion_matrix(y_te, y_pred)

lr_metrics, lr_cm = evaluate_model("Logistic Regression", lr_model,
                                    X_train_sc, y_train, X_test_sc, y_test)
rf_metrics, rf_cm = evaluate_model("Random Forest", rf_model,
                                    X_train_sc, y_train, X_test_sc, y_test)

# ── Plot Perbandingan Metrik ─────────────────────────────────
metrics_df = pd.DataFrame([lr_metrics, rf_metrics]).set_index("Model")
plot_cols   = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
x           = np.arange(len(plot_cols))
width       = 0.35

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Perbandingan Performa Model – Klasifikasi Stunting", fontsize=13, fontweight="bold")

bars1 = axes[0].bar(x - width/2, metrics_df.loc["Logistic Regression", plot_cols], width,
                    label="Logistic Regression", color="#3498db", edgecolor="white")
bars2 = axes[0].bar(x + width/2, metrics_df.loc["Random Forest", plot_cols], width,
                    label="Random Forest", color="#e67e22", edgecolor="white")
axes[0].set_xticks(x); axes[0].set_xticklabels(plot_cols, rotation=15)
axes[0].set_ylim(0, 1.15)
axes[0].set_ylabel("Skor")
axes[0].set_title("Metrik Evaluasi")
axes[0].legend()
axes[0].set_axisbelow(True); axes[0].grid(axis="y", alpha=0.3)
for bar in list(bars1) + list(bars2):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.2f}", ha="center", fontsize=8)

# Confusion matrix RF
sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
            xticklabels=["Tidak Stunting","Stunting"],
            yticklabels=["Tidak Stunting","Stunting"],
            linewidths=1, linecolor="white")
axes[1].set_title("Confusion Matrix – Random Forest")
axes[1].set_xlabel("Prediksi"); axes[1].set_ylabel("Aktual")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_model_evaluation.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Plot evaluasi disimpan → {OUTPUT_DIR}/03_model_evaluation.png")

# ── Feature Importance ───────────────────────────────────────
fi_df = pd.DataFrame({
    "Fitur"      : FEATURE_NAMES,
    "Importance" : rf_model.feature_importances_
}).sort_values("Importance", ascending=True)

plt.figure(figsize=(9, 5))
colors_fi = ["#e74c3c" if v > 0.15 else "#3498db" for v in fi_df["Importance"]]
bars = plt.barh(fi_df["Fitur"], fi_df["Importance"], color=colors_fi, edgecolor="white")
plt.xlabel("Feature Importance (Mean Decrease Impurity)")
plt.title("Feature Importance – Random Forest\n(Merah = Faktor Utama Risiko Stunting)", fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"✓ Feature importance disimpan → {OUTPUT_DIR}/04_feature_importance.png")

print("\nTop 3 faktor penyebab risiko stunting:")
for _, row in fi_df.sort_values("Importance", ascending=False).head(3).iterrows():
    print(f"   → {row['Fitur']:<22} : {row['Importance']:.4f}")

# ══════════════════════════════════════════════════════════
# BAGIAN 5 — SIMPAN MODEL & ARTEFAK
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  BAGIAN 5: SIMPAN MODEL")
print("=" * 60)

joblib.dump(rf_model,    f"{MODEL_DIR}/random_forest.pkl")
joblib.dump(lr_model,    f"{MODEL_DIR}/logistic_regression.pkl")
joblib.dump(scaler,      f"{MODEL_DIR}/scaler.pkl")
joblib.dump(le_gender,   f"{MODEL_DIR}/le_gender.pkl")
joblib.dump(le_wasting,  f"{MODEL_DIR}/le_wasting.pkl")
joblib.dump(le_target,   f"{MODEL_DIR}/le_target.pkl")

metrics_df.to_csv(f"{OUTPUT_DIR}/model_metrics.csv")

print("✓ Model tersimpan di folder 'models/'")
print("✓ Metrik tersimpan di 'output/model_metrics.csv'")
print("\n✅ Training pipeline selesai. Jalankan: streamlit run dashboard.py")
