# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import numpy as np
# import joblib

# app = Flask(__name__)
# CORS(app)  # Enable CORS for all routes

# # Load the pre-trained VotingClassifier (ensemble model)
# voting_clf = joblib.load('./models/ensemble.pkl')

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     features = np.array(data['features']).reshape(1, -1)

#     # Make prediction using the ensemble model
#     prediction = voting_clf.predict(features)
#     risk = 'High' if prediction[0] == 1 else 'Low'

#     return jsonify({'risk': risk})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os

app = Flask(__name__)

# Enable CORS only for the '/predict' route
CORS(app, resources={r"/predict": {"origins": "*"}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the directory of app.py
MODEL_PATH = os.path.join(BASE_DIR, "../models/ensemble.pkl")

try:
    voting_clf = joblib.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    voting_clf = None  # Set to None to avoid crashes

@app.route('/')
def home():
    return jsonify({"message": "Flask API is running!", "status": "OK"})

@app.route('/predict', methods=['POST'])
def predict():
    if voting_clf is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        if "features" not in data:
            return jsonify({"error": "Missing 'features' key in request JSON"}), 400
        
        features = np.array(data["features"]).reshape(1, -1)

        # Make prediction using the ensemble model
        prediction = voting_clf.predict(features)
        risk = "High" if prediction[0] == 1 else "Low"

        return jsonify({"risk": risk})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
