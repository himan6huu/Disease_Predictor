import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ================= LOAD DATA =================
data = pd.read_csv("Data/Training.csv")

data.columns = data.columns.str.replace(r"\.\d+$", "", regex=True)
data = data.loc[:, ~data.columns.duplicated()]

# Remove diseases with very few samples
data = data.groupby("prognosis").filter(lambda x: len(x) > 10)

X = data.iloc[:, :-1]
y = data["prognosis"]

# ================= ENCODE LABELS =================
le = LabelEncoder()
y_enc = le.fit_transform(y)

# ================= SPLIT DATA =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)

# ================= TRAIN MODEL =================
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=None,
    random_state=42
)

model.fit(X_train, y_train)

# ================= EVALUATE =================
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy*100:.2f}%")

# ================= SAVE MODEL =================
joblib.dump(model, "model/disease_model.pkl")
joblib.dump(le, "model/label_encoder.pkl")

print("✅ Model saved successfully")