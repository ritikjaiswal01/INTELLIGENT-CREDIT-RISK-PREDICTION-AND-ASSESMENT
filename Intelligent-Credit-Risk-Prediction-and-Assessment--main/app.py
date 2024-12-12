from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = "models/credit_risk_model.pkl"
model = joblib.load(model_path)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = np.array([[
            data["age"],
            data["income"],
            data["credit_score"],
            data["loan_amount"],
            data["debt_to_income"],
            data["existing_loan"],
        ]])

        # Perform prediction
        prediction = model.predict(features)[0]
        result = "Eligible" if prediction == 0 else "Not Eligible"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
