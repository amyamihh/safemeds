import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from itertools import combinations
import random

# -------------------- LOAD DATA --------------------
df = pd.read_csv("dataset/db_drug_interactions.csv")
df = df[["Drug 1", "Drug 2"]]
df["label"] = 1

# Use strong harmful sample
df_positive = df.sample(n=120000, random_state=42)

harmful_pairs = set(
    tuple(sorted([row["Drug 1"], row["Drug 2"]]))
    for _, row in df_positive.iterrows()
)

# -------------------- GENERATE SAFE PAIRS --------------------
drugs = list(set(df["Drug 1"]).union(set(df["Drug 2"])))
all_possible_pairs = set(tuple(sorted(pair)) for pair in combinations(drugs, 2))
safe_pairs = list(all_possible_pairs - harmful_pairs)

# Controlled safe sampling (balanced but slightly reduced noise)
safe_sample = random.sample(safe_pairs, 50000)

df_harmful = pd.DataFrame(list(harmful_pairs), columns=["Drug 1", "Drug 2"])
df_harmful["label"] = 1

df_safe = pd.DataFrame(safe_sample, columns=["Drug 1", "Drug 2"])
df_safe["label"] = 0

df_final = pd.concat([df_harmful, df_safe], ignore_index=True)

# -------------------- FEATURE ENGINEERING --------------------
harmful_df = df_final[df_final["label"] == 1]

drug1_harm = harmful_df["Drug 1"].value_counts()
drug2_harm = harmful_df["Drug 2"].value_counts()

drug1_total = df_final["Drug 1"].value_counts()
drug2_total = df_final["Drug 2"].value_counts()

df_final["drug1_harm_degree"] = df_final["Drug 1"].map(drug1_harm).fillna(0)
df_final["drug2_harm_degree"] = df_final["Drug 2"].map(drug2_harm).fillna(0)

df_final["drug1_total_degree"] = df_final["Drug 1"].map(drug1_total).fillna(0)
df_final["drug2_total_degree"] = df_final["Drug 2"].map(drug2_total).fillna(0)

# 🔥 NEW: Interaction strength ratios (accuracy booster)
df_final["interaction_ratio_1"] = (
    df_final["drug1_harm_degree"] / (df_final["drug1_total_degree"] + 1)
)

df_final["interaction_ratio_2"] = (
    df_final["drug2_harm_degree"] / (df_final["drug2_total_degree"] + 1)
)
print(df_final.columns)

# -------------------- ADD ADDITIONAL FEATURES --------------------
df_final["dosage"] = [random.randint(100, 1000) for _ in range(len(df_final))]
df_final["age"] = [random.randint(18, 80) for _ in range(len(df_final))]
df_final["gender"] = [random.randint(0, 1) for _ in range(len(df_final))]
df_final["disease_flag"] = [random.randint(0, 1) for _ in range(len(df_final))]
# Chemical similarity (simulate molecular similarity)
df_final["chemical_similarity"] = [random.uniform(0.2, 0.9) for _ in range(len(df_final))]

# Side effect score (simulate side effect severity)
df_final["side_effect_score"] = [random.uniform(0.1, 1.0) for _ in range(len(df_final))]

# Toxicity score (simulate toxicity level)
df_final["toxicity_score"] = [random.uniform(0.1, 1.0) for _ in range(len(df_final))]

# -------------------- TRAIN TEST SPLIT --------------------
X = df_final[['Drug 1', 'Drug 2',
        'interaction_ratio_1', 'interaction_ratio_2',
        'drug1_harm_degree', 'drug2_harm_degree',
        'drug1_total_degree', 'drug2_total_degree',
        'chemical_similarity',
        'side_effect_score',
        'toxicity_score']]

y = df_final["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------- CATBOOST MODEL --------------------
model = CatBoostClassifier(
    iterations=1300,
    learning_rate=0.015,
    depth=10,
    l2_leaf_reg=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=100
)

model.fit(X_train, y_train, cat_features=[0, 1])

# -------------------- EVALUATION --------------------
y_pred = model.predict(X_test)

print("\nFinal Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
model.save_model("safemeds_model.cbm")
print("Model saved successfully.")
