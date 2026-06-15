import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ======================================
# LOAD DATA
# ======================================

print("Loading Dataset...")

df = pd.read_excel("../data/Overall Data.xlsx")

print("Jumlah Data Awal :", len(df))

# ======================================
# DATA CLEANING
# ======================================

df["Age (Month)"] = pd.to_numeric(
    df["Age (Month)"],
    errors="coerce"
)

df["Weight"] = pd.to_numeric(
    df["Weight"],
    errors="coerce"
)

df["Height"] = pd.to_numeric(
    df["Height"],
    errors="coerce"
)

df = df.dropna()

print("Jumlah Data Setelah Cleaning :", len(df))

# ======================================
# FEATURE & TARGET
# ======================================

X = df[
    [
        "Gender",
        "Age (Month)",
        "Weight",
        "Height"
    ]
].copy()

y = df["Height for Age"]

# ======================================
# ENCODING
# ======================================

gender_encoder = LabelEncoder()

X["Gender"] = gender_encoder.fit_transform(
    X["Gender"]
)

target_encoder = LabelEncoder()

y = target_encoder.fit_transform(y)

# ======================================
# TRAIN TEST SPLIT
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData Train :", len(X_train))
print("Data Test :", len(X_test))

# ======================================
# PERBANDINGAN MODEL
# ======================================

print("\n===== PERBANDINGAN MODEL =====")

models = {

    "Random Forest":
        RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        )
}

best_model = None
best_accuracy = 0

for name, clf in models.items():

    clf.fit(X_train, y_train)

    pred = clf.predict(X_test)

    acc = accuracy_score(
        y_test,
        pred
    )

    print(f"{name} : {acc:.4f}")

    if acc > best_accuracy:

        best_accuracy = acc
        best_model = clf

print("\nModel Terbaik :", best_model.__class__.__name__)
print("Accuracy :", best_accuracy)

# ======================================
# TRAIN FINAL MODEL
# ======================================

model = best_model

# ======================================
# PREDIKSI
# ======================================

y_pred = model.predict(X_test)

# ======================================
# PROBABILITY
# ======================================

if hasattr(model, "predict_proba"):

    y_prob = model.predict_proba(
        X_test
    )[:, 1]

    auc = roc_auc_score(
        y_test,
        y_prob
    )

else:

    auc = 0

# ======================================
# CROSS VALIDATION
# ======================================

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

# ======================================
# EVALUASI
# ======================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n===== HASIL EVALUASI =====")

print("Accuracy :", accuracy)

print("Precision :", precision)

print("Recall :", recall)

print("F1 Score :", f1)

print("ROC AUC :", auc)

print(
    "Cross Validation Mean :",
    cv_scores.mean()
)

print(
    "\nClassification Report\n"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# ======================================
# FEATURE IMPORTANCE
# ======================================

print("\n===== FEATURE IMPORTANCE =====")

if hasattr(
    model,
    "feature_importances_"
):

    for feature, score in zip(
        X.columns,
        model.feature_importances_
    ):

        print(
            f"{feature} : {score:.4f}"
        )

# ======================================
# SAVE METRICS
# ======================================

metrics = {

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1": f1,

    "roc_auc": auc,

    "cv_mean": cv_scores.mean(),

    "cm": cm,

    "feature_importance":

        model.feature_importances_

        if hasattr(
            model,
            "feature_importances_"
        )

        else None
}

# ======================================
# SAVE FILE
# ======================================

joblib.dump(
    model,
    "model.pkl"
)

joblib.dump(
    target_encoder,
    "target_encoder.pkl"
)

joblib.dump(
    gender_encoder,
    "gender_encoder.pkl"
)

joblib.dump(
    metrics,
    "metrics.pkl"
)

print(
    "\nModel berhasil disimpan"
)

print(
    "File:"
)

print("- model.pkl")
print("- metrics.pkl")
print("- target_encoder.pkl")
print("- gender_encoder.pkl")