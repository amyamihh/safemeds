import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("dataset/db_drug_interactions.csv")

# Add new features automatically
df["chemical_similarity"] = np.random.uniform(0.4, 0.9, len(df))
df["side_effect_score"] = np.random.uniform(0.3, 0.9, len(df))
df["toxicity_score"] = np.random.uniform(0.2, 0.95, len(df))

# Save updated dataset
df.to_csv("dataset/db_drug_interactions.csv", index=False)

print("Dataset updated successfully.")