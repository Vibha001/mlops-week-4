import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)

# Load dataset
data = pd.read_csv("data/iris.csv")

# Prepare features and target
X = data.drop(columns=["species"])
y = data["species"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train model
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Save model
joblib.dump(model, "artifacts/model.joblib")

# Save accuracy to metrics.csv
with open("metrics.csv", "w") as f:
    f.write("accuracy\n")
    f.write(f"{acc:.4f}\n")

print(f"Model trained. Accuracy: {acc:.4f}")
