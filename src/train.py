import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from config import MODEL_PATH, DATA_DIR, OUTPUTS_DIR

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from xgboost import XGBClassifier


# ============================================================
# AI FRAUD DETECTION SYSTEM
# STEP 6 — MODEL SAVING + REAL PREDICTION
# ============================================================


# ============================================================
# 1. CREATE DATASET
# ============================================================

np.random.seed(42)

n_transactions = 10_000

data = {
    "amount": np.random.uniform(
        5, 2000, n_transactions
    ),

    "transaction_hour": np.random.randint(
        0, 24, n_transactions
    ),

    "distance_from_home": np.random.uniform(
        0, 500, n_transactions
    ),

    "merchant_risk": np.random.uniform(
        0, 1, n_transactions
    ),

    "device_trusted": np.random.randint(
        0, 2, n_transactions
    ),

    "previous_transactions": np.random.randint(
        0, 20, n_transactions
    ),
}

df = pd.DataFrame(data)


# ============================================================
# 2. CREATE FRAUD LABEL
# ============================================================

fraud_score = (
    (df["amount"] > 1500) * 2
    + (df["transaction_hour"].between(0, 5)) * 2
    + (df["distance_from_home"] > 300) * 2
    + (df["merchant_risk"] > 0.7) * 2
    + (df["device_trusted"] == 0) * 1
)

random_noise = np.random.random(
    n_transactions
)

df["is_fraud"] = (
    (fraud_score + random_noise > 4)
    .astype(int)
)


DATA_DIR.mkdir(exist_ok=True)
df.to_csv(DATA_DIR / "transactions.csv", index=False)
print(f"Dataset saved to: {DATA_DIR / 'transactions.csv'}")


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

features = [
    "amount",
    "transaction_hour",
    "distance_from_home",
    "merchant_risk",
    "device_trusted",
    "previous_transactions"
]

X = df[features]

y = df["is_fraud"]


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

print("=" * 70)
print("AI FRAUD DETECTION SYSTEM")
print("STEP 6 — PRODUCTION MODEL")
print("=" * 70)

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Model trained successfully!")


# ============================================================
# 6. EVALUATE MODEL
# ============================================================

predictions = model.predict(
    X_test
)

probabilities = model.predict_proba(
    X_test
)[:, 1]

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1-Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc:.4f}"
)


# ============================================================
# 7. SAVE MODEL
# ============================================================

model_file = MODEL_PATH
MODEL_PATH.parent.mkdir(exist_ok=True)

joblib.dump(
    model,
    model_file
)

print("\n" + "=" * 70)
print("MODEL SAVED")
print("=" * 70)

print(
    f"\nModel saved as: {model_file}"
)


# ============================================================
# 8. LOAD MODEL AGAIN
# ============================================================

loaded_model = joblib.load(
    model_file
)

print("\nModel loaded successfully!")


# ============================================================
# 9. CREATE A NEW TRANSACTION
# ============================================================

new_transaction = pd.DataFrame({
    "amount": [1750],
    "transaction_hour": [2],
    "distance_from_home": [420],
    "merchant_risk": [0.85],
    "device_trusted": [0],
    "previous_transactions": [2]
})


print("\n" + "=" * 70)
print("NEW TRANSACTION")
print("=" * 70)

print()

print(
    new_transaction.to_string(
        index=False
    )
)


# ============================================================
# 10. PREDICT NEW TRANSACTION
# ============================================================

prediction = loaded_model.predict(
    new_transaction
)[0]

probability = loaded_model.predict_proba(
    new_transaction
)[0][1]


# ============================================================
# 11. DISPLAY AI DECISION
# ============================================================

print("\n" + "=" * 70)
print("AI FRAUD ANALYSIS")
print("=" * 70)

print(
    f"\nFraud probability: {probability * 100:.2f}%"
)

if prediction == 1:

    print("\n⚠️ FRAUD DETECTED")

    print(
        "The transaction has been classified "
        "as potentially fraudulent."
    )

else:

    print("\n✅ TRANSACTION APPEARS NORMAL")

    print(
        "The transaction has been classified "
        "as normal."
    )


# ============================================================
# 12. MODEL FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print()

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# 13. FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(9, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.gca().invert_yaxis()

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel(
    "Importance"
)

plt.tight_layout()

OUTPUTS_DIR.mkdir(exist_ok=True)
plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150, bbox_inches="tight")

plt.show()


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

plt.figure(figsize=(7, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Fraud"
    ]
).plot()

plt.title(
    "Fraud Detection - Confusion Matrix"
)

plt.tight_layout()

plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")

plt.show()


# ============================================================
# 15. PROJECT FILES
# ============================================================

print("\n" + "=" * 70)
print("PROJECT FILES")
print("=" * 70)

print()

print(
    "Model file exists:",
    os.path.exists(model_file)
)

print(
    "Model path:",
    os.path.abspath(model_file)
)


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nThe system can now:")

print("✓ Train a fraud detection model")
print("✓ Evaluate the model")
print("✓ Save the trained model")
print("✓ Load the trained model")
print("✓ Analyze new transactions")
print("✓ Calculate fraud probability")
print("✓ Detect potentially fraudulent transactions")
print("✓ Explain feature importance")

print("\nNext step:")
print("Build a reusable prediction API with FastAPI.")

print("\nProcess completed successfully.")