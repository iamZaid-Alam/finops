#!/usr/bin/env python3
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Hyper‑parameters from env
N_ESTIMATORS = int(os.getenv("N_ESTIMATORS", 100))
MAX_DEPTH    = int(os.getenv("MAX_DEPTH", 5))
TEST_SPLIT   = float(os.getenv("TEST_SPLIT", 0.2))

print(f"[INFO] n_estimators={N_ESTIMATORS}, max_depth={MAX_DEPTH}, test_split={TEST_SPLIT}")

# Load CSV (relative path works both locally & in cloud runset)
DATA_PATH = "iris.csv"
if not os.path.isfile(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found")

df = pd.read_csv(DATA_PATH)

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Optional encode if target is string
if y.dtype == object:
    y = y.astype("category").cat.codes

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    max_depth=MAX_DEPTH,
    random_state=42,
)
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"[METRIC] accuracy={acc:.4f}")

# Save artefact
os.makedirs("/opt/output", exist_ok=True)
joblib.dump(clf, "app/iris_model.joblib")
print("[INFO] model saved to app/iris_model.joblib")
