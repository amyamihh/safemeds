import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load dataset
data = pd.read_csv("dataset/db_drug_interactions.csv")

# Combine drugs into sequence
data["drug_pair"] = data["Drug 1"] + " " + data["Drug 2"]

# Encode drugs
encoder = LabelEncoder()
all_drugs = list(data["Drug 1"]) + list(data["Drug 2"])
encoder.fit(all_drugs)

data["drug1_enc"] = encoder.transform(data["Drug 1"])
data["drug2_enc"] = encoder.transform(data["Drug 2"])

# Create sequences
X = data[["drug1_enc", "drug2_enc"]].values
y = np.random.randint(0,2,len(X))   # placeholder labels

# LSTM model
model = Sequential()
model.add(Embedding(input_dim=len(encoder.classes_), output_dim=16, input_length=2))
model.add(LSTM(32))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train model
model.fit(X, y, epochs=5, batch_size=32)

# Save model
model.save("lstm_model.h5")

print("LSTM model trained and saved.")