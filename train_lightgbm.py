import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset/db_drug_interactions.csv")
df = df[["Drug 1", "Drug 2"]]
df["label"] = 1

# Sample harmful pairs
df_positive = df.sample(n=100000, random_state=42)

# Remove duplicate order pairs
harmful_pairs = set(
    tuple(sorted([row["Drug 1"], row["Drug 2"]]))
    for _, row in df_positive.iterrows()
)

# Get unique drugs
drugs = list(set(df["Drug 1"]).union(set(df["Drug 2"])))

from itertools import combinations
all_possible_pairs = set(tuple(sorted(pair)) for pair in combinations(drugs, 2))

safe_pairs = list(all_possible_pairs - harmful_pairs)

import random
safe_sample = random.sample(safe_pairs, 100000)

# Create final dataset
df_harmful = pd.DataFrame(list(harmful_pairs), columns=["Drug 1", "Drug 2"])
df_harmful["label"] = 1

df_safe = pd.DataFrame(safe_sample, columns=["Drug 1", "Drug 2"])
df_safe["label"] = 0

df_final = pd.concat([df_harmful, df_safe], ignore_index=True)

# ----- ADD DEGREE FEATURES -----
harmful_df = df_final[df_final["label"] == 1]

drug1_counts = harmful_df["Drug 1"].value_counts()
drug2_counts = harmful_df["Drug 2"].value_counts()

df_final["drug1_degree"] = df_final["Drug 1"].map(drug1_counts).fillna(0)
df_final["drug2_degree"] = df_final["Drug 2"].map(drug2_counts).fillna(0)

# Prepare features
X = df_final[["Drug 1", "Drug 2", "drug1_degree", "drug2_degree"]]
y = df_final["label"]

X["Drug 1"] = X["Drug 1"].astype("category")
X["Drug 2"] = X["Drug 2"].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train LightGBM
model = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
