# tests/test_model.py

import joblib
import pandas as pd

def test_model_accuracy():
    model = joblib.load("artifacts/model.joblib")
    data = pd.read_csv("data/iris.csv")
    X = data.drop(columns=["species"])
    y = data["species"]
    acc = model.score(X, y)
    assert acc > 0.90, f"Model accuracy too low: {acc}"
