import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Regression.linear_regresssion import LinearRegression
from Regression.logistic_Regression import LogisticRegression

# ══════════════════════════════════════════════════════════════════
# CHANGE THESE VALUES TO TEST YOURSELF
# ══════════════════════════════════════════════════════════════════

my_age            = 30
my_blood_pressure = 100
my_cholesterol    = 160
my_heart_rate     = 60

my_sepal_length = 5.0
my_sepal_width  = 3.4
my_petal_length = 1.5
my_petal_width  = 0.2

# ══════════════════════════════════════════════════════════════════
# DO NOT CHANGE ANYTHING BELOW THIS LINE
# ══════════════════════════════════════════════════════════════════

base_path = os.path.dirname(__file__)

# ── Train Linear Regression ───────────────────────────────────────
df = pd.read_excel(os.path.join(base_path, "test data", "regression", "heart_disease.xlsx"))
X  = df[["age", "blood_pressure", "cholesterol", "heart_rate"]].values
z  = df["risk_score"].values

model_lr = LinearRegression()
model_lr.fit(X, z)

# ── Single Prediction ─────────────────────────────────────────────
patient = np.array([[my_age, my_blood_pressure, my_cholesterol, my_heart_rate]])
risk    = model_lr.predict(patient)[0]

print("=" * 45)
print("HEART DISEASE RISK RESULT")
print("=" * 45)
print(f"Age:            {my_age}")
print(f"Blood Pressure: {my_blood_pressure}")
print(f"Cholesterol:    {my_cholesterol}")
print(f"Heart Rate:     {my_heart_rate}")
print(f"Risk Score:     {risk:.1f} / 100")
if risk < 60:
    print("Assessment:     LOW RISK")
elif risk < 75:
    print("Assessment:     MEDIUM RISK")
else:
    print("Assessment:     HIGH RISK")

# ── Accuracy on unseen heart disease data ─────────────────────────
unseen_patients = np.array([
    [30, 100, 160, 60],   # young healthy
    [50, 140, 220, 80],   # middle aged
    [70, 170, 280, 95],   # high risk
    [45, 120, 190, 72],   # medium
    [60, 155, 250, 88],   # high
    [35, 110, 175, 65],   # low
    [55, 145, 235, 82],   # medium
    [65, 165, 265, 91],   # high
])

# true risk scores calculated from same formula: 0.4*age + 0.2*bp + 0.1*chol + 0.05*hr
true_risks = np.array([
    0.4*30  + 0.2*100 + 0.1*160 + 0.05*60,
    0.4*50  + 0.2*140 + 0.1*220 + 0.05*80,
    0.4*70  + 0.2*170 + 0.1*280 + 0.05*95,
    0.4*45  + 0.2*120 + 0.1*190 + 0.05*72,
    0.4*60  + 0.2*155 + 0.1*250 + 0.05*88,
    0.4*35  + 0.2*110 + 0.1*175 + 0.05*65,
    0.4*55  + 0.2*145 + 0.1*235 + 0.05*82,
    0.4*65  + 0.2*165 + 0.1*265 + 0.05*91,
])

predicted_risks = model_lr.predict(unseen_patients).flatten()
mse = np.mean((predicted_risks - true_risks) ** 2)

print(f"\n{'─' * 45}")
print("ACCURACY ON UNSEEN PATIENTS")
print(f"{'─' * 45}")
print(f"{'Age':<6} {'BP':<6} {'Chol':<6} {'HR':<6} {'Predicted':<12} {'Actual'}")
print("-" * 45)
for p, pred, true in zip(unseen_patients, predicted_risks, true_risks):
    print(f"{p[0]:<6} {p[1]:<6} {p[2]:<6} {p[3]:<6} {pred:<12.1f} {true:.1f}")
print(f"\nMSE on unseen data: {mse:.4f}")
print(f"(lower is better — 0.0 is perfect)")

# ── Train Logistic Regression ─────────────────────────────────────
df      = pd.read_excel(os.path.join(base_path, "test data", "regression", "iris_flowers.xlsx"))
X       = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
species = df["species"].values
le      = LabelEncoder()
z       = le.fit_transform(species)

model_log = LogisticRegression()
model_log.fit(X, z)

# ── Single Prediction ─────────────────────────────────────────────
flower    = np.array([[my_sepal_length, my_sepal_width, my_petal_length, my_petal_width]])
pred      = model_log.predict(flower)
pred_name = le.inverse_transform(pred)[0]

print("\n" + "=" * 45)
print("IRIS FLOWER RESULT")
print("=" * 45)
print(f"Sepal Length: {my_sepal_length} cm")
print(f"Sepal Width:  {my_sepal_width} cm")
print(f"Petal Length: {my_petal_length} cm")
print(f"Petal Width:  {my_petal_width} cm")
print(f"Predicted Species: {pred_name}")

# ── Accuracy on unseen flower data ────────────────────────────────
unseen_flowers = np.array([
    [5.1, 3.5, 1.4, 0.2],   # Setosa
    [4.9, 3.1, 1.5, 0.1],   # Setosa
    [6.0, 2.9, 4.5, 1.5],   # Versicolor
    [5.8, 2.7, 4.1, 1.0],   # Versicolor
    [6.7, 3.1, 5.6, 2.4],   # Virginica
    [6.4, 3.2, 5.3, 2.3],   # Virginica
    [5.2, 3.4, 1.4, 0.2],   # Setosa
    [6.1, 2.8, 4.7, 1.2],   # Versicolor
])
true_species  = ["Setosa", "Setosa", "Versicolor", "Versicolor",
                 "Virginica", "Virginica", "Setosa", "Versicolor"]
true_encoded  = le.transform(true_species)

preds         = model_log.predict(unseen_flowers)
pred_names    = le.inverse_transform(preds)
accuracy      = np.mean(preds == true_encoded)

print(f"\n{'─' * 45}")
print("ACCURACY ON UNSEEN FLOWERS")
print(f"{'─' * 45}")
print(f"{'Predicted':<15} {'Expected':<15} {'Correct'}")
print("-" * 45)
for pred_name, true_name in zip(pred_names, true_species):
    match = "✅" if pred_name == true_name else "❌"
    print(f"{pred_name:<15} {true_name:<15} {match}")
print(f"\nAccuracy on unseen data: {accuracy*100:.1f}%")
print(f"(higher is better — 100% is perfect)")
