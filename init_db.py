import sqlite3
import pandas as pd

# Load CSV
df = pd.read_csv("dataset/db_drug_interactions.csv")

# Connect to SQLite database (creates file if not exists)
conn = sqlite3.connect("safemeds.db")

# Save to database table
df.to_sql("drug_interactions", conn, if_exists="replace", index=False)

conn.close()

print("Database created successfully as safemeds.db")
