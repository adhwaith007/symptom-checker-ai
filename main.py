# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 22:36:44 2025

@author: bijip
"""
from flask import Flask, request, jsonify
from model import DiseasePredictor

app = Flask(__name__)
predictor = DiseasePredictor()
predictor.load_and_train()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Disease prediction API is ready."})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "symptoms" not in data:
        return jsonify({"error": "Invalid input. Expected JSON with 'symptoms' key."}), 400

    predictions = predictor.predict(data["symptoms"])
    return jsonify({"predicted_disease": predictions})

@app.route("/test", methods=["GET"])
def test():
    return jsonify({"message": "Test route is working"})

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)


