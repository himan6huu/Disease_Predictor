from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from difflib import get_close_matches

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ================= DATABASE CONFIG =================
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ================= USER MODEL =================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)





class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    probability = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()
# ================= LOAD ML MODEL =================
model = joblib.load("model/disease_model.pkl")
le = joblib.load("model/label_encoder.pkl")

# Load symptom list
data = pd.read_csv("Data/Training.csv")
data.columns = data.columns.str.replace(r"\.\d+$", "", regex=True)
data = data.loc[:, ~data.columns.duplicated()]
symptom_list = list(data.columns[:-1])

# Example disease info dictionary
disease_info = {
    "Flu": {
        "description": "Influenza is a viral infection affecting respiratory system.",
        "precautions": "Rest, hydration, paracetamol, consult doctor if severe."
    },
    "Malaria": {
        "description": "Mosquito-borne infectious disease.",
        "precautions": "Use mosquito nets, take antimalarial medication."
    }
}

# ================= ROUTES =================

@app.route("/")
def index_redirect():
    if "user" in session:
        return redirect(url_for("home"))
    return redirect(url_for("login"))

# -------- SIGNUP --------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        password = generate_password_hash(request.form["password"])

        if User.query.filter_by(username=username).first():
            return "Username already exists"

        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("signup.html")

# -------- LOGIN --------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return "Invalid credentials"

    return render_template("login.html")

# -------- HOME / DASHBOARD --------

@app.route("/home", methods=["GET", "POST"])
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    results = []
    description = None
    precautions = None

    accuracy_value = 92  # You can calculate dynamically later

    # Get current user
    current_user = User.query.filter_by(username=session["user"]).first()

# 🔥 Safety check
    # 🔥 Safety check
    if not current_user:
        session.pop("user", None)
        return redirect(url_for("login"))

    if request.method == "POST":

        # Get selected symptoms
        user_input = request.form.getlist("symptoms")
        input_data = np.zeros(len(symptom_list))

        for symptom in user_input:
            symptom = symptom.strip().lower()
            match = get_close_matches(symptom, symptom_list, n=1, cutoff=0.6)
            if match:
                index = symptom_list.index(match[0])
                input_data[index] = 1

        # Model prediction
        probs = model.predict_proba([input_data])[0]
        top3_idx = probs.argsort()[-3:][::-1]

        for idx in top3_idx:
            disease = le.inverse_transform([idx])[0]
            probability = round(probs[idx] * 100, 2)
            results.append((disease, probability))

        # Save top prediction to database
        if results:
            new_prediction = Prediction(
                user_id=current_user.id,
                disease=results[0][0],
                probability=results[0][1]
            )

            db.session.add(new_prediction)
            db.session.commit()

    # ===== ALWAYS FETCH USER HISTORY =====
    user_predictions = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(Prediction.timestamp.desc()).all()

    total_predictions = len(user_predictions)

    if total_predictions > 0:
        last_disease = user_predictions[0].disease
        top_prob = user_predictions[0].probability
    else:
        last_disease = "None"
        top_prob = 0

    # Risk calculation
    if top_prob > 80:
        risk_level = "High"
        risk_color = "danger"
    elif top_prob > 50:
        risk_level = "Medium"
        risk_color = "warning"
    else:
        risk_level = "Low"
        risk_color = "success"

    # Disease info
    info = disease_info.get(last_disease, {})
    description = info.get("description")
    precautions = info.get("precautions")

    return render_template(
        "index.html",
        username=session["user"],
        results=results,
        description=description,
        precautions=precautions,
        symptom_list=symptom_list,
        total_predictions=total_predictions,
        last_disease=last_disease,
        accuracy_value=accuracy_value,
        risk_level=risk_level,
        risk_color=risk_color,
        user_predictions=user_predictions
    )

# -------- LOGOUT --------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# ================= RUN APP =================
if __name__ == "__main__":
    app.run(debug=True)