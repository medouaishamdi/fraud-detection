import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt

print("=" * 70)
print("AI FRAUD DETECTION - EXPLAINABLE AI")
print("=" * 70)

# ============================================================
# 1. LOAD TRAINED MODEL
# ============================================================

from config import MODEL_PATH, OUTPUTS_DIR
OUTPUTS_DIR.mkdir(exist_ok=True)

print("\nLoading trained model...")

try:
    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully!")
    print("Model type:", type(model).__name__)

except Exception as e:
    print("\nERROR loading model:")
    print(e)
    print("\nRun `python src/train.py` first to create models/fraud_model.pkl.")
    exit()


# ============================================================
# 2. MODEL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print("Model:", type(model).__name__)

if hasattr(model, "n_estimators"):
    print("Number of trees:", model.n_estimators)

if hasattr(model, "max_depth"):
    print("Maximum depth:", model.max_depth)

if hasattr(model, "classes_"):
    print("Classes:", model.classes_)


# ============================================================
# 3. FEATURES
# ============================================================

feature_names = [
    "amount",
    "transaction_hour",
    "distance_from_home",
    "merchant_risk",
    "device_trusted",
    "previous_transactions"
]


# ============================================================
# 4. CREATE NORMAL TRANSACTION
# ============================================================

normal_transaction = pd.DataFrame(
    [[
        300.0,
        14,
        5.0,
        0.10,
        1,
        15
    ]],
    columns=feature_names
)


# ============================================================
# 5. CREATE SUSPICIOUS TRANSACTION
# ============================================================

fraud_transaction = pd.DataFrame(
    [[
        1800.0,
        3,
        400.0,
        0.90,
        0,
        1
    ]],
    columns=feature_names
)


# ============================================================
# 6. CREATE SHAP EXPLAINER
# ============================================================

print("\n" + "=" * 70)
print("CREATING SHAP EXPLAINER")
print("=" * 70)

try:

    explainer = shap.TreeExplainer(model)

    print("SHAP TreeExplainer created successfully!")

except Exception as e:

    print("\nERROR creating SHAP explainer:")
    print(e)

    print("\nThe saved model may not be a tree-based model.")
    exit()


# ============================================================
# 7. NORMAL TRANSACTION PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("NORMAL TRANSACTION ANALYSIS")
print("=" * 70)

print("\nTransaction:")
print(normal_transaction.to_string(index=False))

normal_prediction = model.predict(normal_transaction)[0]

normal_probability = model.predict_proba(
    normal_transaction
)[0][1]

print("\nPrediction:")

if normal_prediction == 1:
    print("🚨 FRAUD")
else:
    print("✅ NORMAL")

print(
    f"Fraud probability: "
    f"{normal_probability * 100:.2f}%"
)


# ============================================================
# 8. FRAUD TRANSACTION PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("SUSPICIOUS TRANSACTION ANALYSIS")
print("=" * 70)

print("\nTransaction:")
print(fraud_transaction.to_string(index=False))

fraud_prediction = model.predict(fraud_transaction)[0]

fraud_probability = model.predict_proba(
    fraud_transaction
)[0][1]

print("\nPrediction:")

if fraud_prediction == 1:
    print("🚨 FRAUD")
else:
    print("✅ NORMAL")

print(
    f"Fraud probability: "
    f"{fraud_probability * 100:.2f}%"
)


# ============================================================
# 9. CALCULATE SHAP VALUES
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING SHAP VALUES")
print("=" * 70)

try:

    shap_explanation = explainer(fraud_transaction)

    values = shap_explanation.values

    print("Original SHAP shape:", values.shape)

    # Some SHAP versions return:
    #
    # (samples, features)
    #
    # while others may return:
    #
    # (samples, features, classes)

    if values.ndim == 3:

        fraud_shap_values = values[0, :, 1]

    elif values.ndim == 2:

        fraud_shap_values = values[0]

    else:

        fraud_shap_values = np.array(values).flatten()

    print(
        "Fraud SHAP values calculated successfully!"
    )

except Exception as e:

    print("\nERROR calculating SHAP values:")
    print(e)

    exit()


# ============================================================
# 10. CREATE EXPLANATION TABLE
# ============================================================

print("\n" + "=" * 70)
print("SHAP FEATURE CONTRIBUTIONS")
print("=" * 70)

explanation = pd.DataFrame({

    "Feature": feature_names,

    "Transaction Value":
        fraud_transaction.iloc[0].values,

    "SHAP Contribution":
        fraud_shap_values

})

explanation["Absolute Contribution"] = (
    explanation["SHAP Contribution"].abs()
)

explanation = explanation.sort_values(
    "Absolute Contribution",
    ascending=False
)

print("\nWhy did the model classify this transaction?")

print(
    explanation[
        [
            "Feature",
            "Transaction Value",
            "SHAP Contribution"
        ]
    ].to_string(index=False)
)


# ============================================================
# 11. HUMAN-READABLE EXPLANATION
# ============================================================

print("\n" + "=" * 70)
print("AI EXPLANATION")
print("=" * 70)

top_features = explanation.head(3)

for _, row in top_features.iterrows():

    feature = row["Feature"]

    value = row["Transaction Value"]

    contribution = row["SHAP Contribution"]

    if contribution > 0:

        direction = (
            "increased the probability of fraud"
        )

    else:

        direction = (
            "decreased the probability of fraud"
        )

    print("\nFeature:", feature)

    print("Value:", value)

    print(
        f"Effect: {direction}"
    )

    print(
        f"SHAP contribution: "
        f"{contribution:.4f}"
    )


# ============================================================
# 12. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("MODEL FEATURE IMPORTANCE")
print("=" * 70)

if hasattr(model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance":
            model.feature_importances_

    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    print(
        importance.to_string(index=False)
    )


# ============================================================
# 13. GENERATE SHAP BAR CHART
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SHAP VISUALIZATION")
print("=" * 70)

plt.figure(figsize=(10, 6))

sorted_explanation = explanation.sort_values(
    "SHAP Contribution"
)

plt.barh(
    sorted_explanation["Feature"],
    sorted_explanation["SHAP Contribution"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel(
    "SHAP Contribution"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "AI Explanation - Fraud Prediction"
)

plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "shap_fraud_explanation.png",
    dpi=150,
    bbox_inches="tight"
)

print(
    "\nSHAP visualization saved as:"
)

print(
    "shap_fraud_explanation.png"
)


# ============================================================
# 14. GENERATE SHAP SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("GENERATING FEATURE IMPORTANCE CHART")
print("=" * 70)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Random Forest Feature Importance"
)

plt.tight_layout()

plt.savefig(
    OUTPUTS_DIR / "feature_importance.png",
    dpi=150,
    bbox_inches="tight"
)

print(
    "Feature importance chart saved as:"
)

print(
    "feature_importance.png"
)


# ============================================================
# 15. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("""
Explainable AI has been added to the fraud detection system.

The system can now:

✓ Load the trained Random Forest model
✓ Analyze transactions
✓ Predict fraud
✓ Calculate fraud probability
✓ Calculate SHAP values
✓ Identify important features
✓ Explain why a transaction is suspicious
✓ Generate SHAP visualization
✓ Generate feature importance visualization

Generated files:

1. shap_fraud_explanation.png
2. feature_importance.png

The AI is now capable of explaining its predictions.
""")

print("=" * 70)