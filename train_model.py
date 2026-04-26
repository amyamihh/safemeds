import pandas as pd
import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ---------------------------
# Load Dataset
# ---------------------------

with open("dataset/DDI database.json") as f:
    data = json.load(f)

interactions_data = data["drug_interactions"]

rows = []

for severity, interactions in interactions_data.items():
    for interaction in interactions:
        rows.append({
            "drug1": interaction["drug_a"],
            "drug2": interaction["drug_b"],
            "risk": severity.capitalize()
        })

df = pd.DataFrame(rows)

# ---------------------------
# Create Text Feature
# ---------------------------

df["combined"] = df["drug1"] + " " + df["drug2"]

X_text = df["combined"]
y = df["risk"]

# ---------------------------
# TF-IDF Vectorization
# ---------------------------

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X_text)

# ---------------------------
# Train/Test Split
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Train Model
# ---------------------------

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ---------------------------
# Evaluation
# ---------------------------

y_pred = model.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------
# Save Model & Vectorizer
# ---------------------------

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel and vectorizer saved successfully!")
