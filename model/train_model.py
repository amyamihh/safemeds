import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("data/drug_interactions.csv")

print("Original Dataset:\n")
print(data.head())

# Create label encoders
le_drug1 = LabelEncoder()
le_drug2 = LabelEncoder()
le_risk = LabelEncoder()

# Convert text to numbers
data["drug1"] = le_drug1.fit_transform(data["drug1"])
data["drug2"] = le_drug2.fit_transform(data["drug2"])
data["interaction_risk"] = le_risk.fit_transform(data["interaction_risk"])

print("\nEncoded Dataset:\n")
print(data.head())
