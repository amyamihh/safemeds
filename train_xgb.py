import pandas as pd
import random
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("dataset/db_drug_interactions.csv")
df = df[["Drug 1", "Drug 2"]]
df["label"] = 1

# Take 100k harmful
df_positive = df.sample(n=100000, random_state=42)

# Create harmful set
harmful_pairs = set(
    tuple(sorted([row["Drug 1"], row["Drug 2"]]))
    for _, row in df_positive.iterrows()
)

# Get unique drugs
drugs = list(set(df["Drug 1"]).union(set(df["Drug 2"])))

# Generate all possible combinations
all_possible_pairs = set(
    tuple(sorted(pair)) for pair in combinations(drugs, 2)
)

# Safe pairs
safe_pairs = list(all_possible_pairs - harmful_pairs)
safe_sample = random.sample(safe_pairs, 100000)

# Convert to DataFrame
df_harmful = pd.DataFrame(list(harmful_pairs), columns=["Drug 1", "Drug 2"])
df_harmful["label"] = 1

df_safe = pd.DataFrame(safe_sample, columns=["Drug 1", "Drug 2"])
df_safe["label"] = 0

# Combine
df_final = pd.concat([df_harmful, df_safe], ignore_index=True)

# Create text
df_final["text"] = df_final["Drug 1"] + " " + df_final["Drug 2"]

X = df_final["text"]
y = df_final["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
vectorizer = TfidfVectorizer(
    ngram_range=(1,2),
    max_features=30000,
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# XGBoost
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
