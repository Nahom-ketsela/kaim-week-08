from flask import Flask, request, jsonify
import logging
import joblib
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define model paths
MODEL_DIR = "models"

# Load models with error handling
models = {"fraud": {}, "credit": {}}

model_filenames = {
    "fraud": {
        "logistic_regression": "logistic_regression_fraud.pkl",
        "decision_tree": "decision_tree_fraud.pkl",
        "random_forest": "random_forest_fraud.pkl",
        "gradient_boosting": "gradient_boosting_fraud.pkl",
    },
    "credit": {
        "logistic_regression": "logistic_regression.pkl",
        "decision_tree": "decision_tree.pkl",
        "random_forest": "random_forest.pkl",
        "gradient_boosting": "gradient_boosting.pkl",
    },
}

# Load models safely
for dataset, model_files in model_filenames.items():
    for model_type, filename in model_files.items():
        try:
            model_path = os.path.join(MODEL_DIR, filename)
            models[dataset][model_type] = joblib.load(model_path)
            logger.info(f"Loaded model: {model_type} for dataset: {dataset}")
        except Exception as e:
            logger.error(f"Error loading model {model_type} ({dataset}): {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to make predictions.
    Expects JSON:
    {
      "dataset": "fraud" or "credit",
      "model_type": "logistic_regression" or "decision_tree" etc.,
      "features": { "col1": val1, "col2": val2, ... }
    }
    """
    try:
        logger.info("Received prediction request")

        # Get JSON data from the request
        data = request.get_json()

        # Validate required fields
        required_fields = {"dataset", "model_type", "features"}
        if not data or not required_fields.issubset(data.keys()):
            logger.error("Missing 'dataset', 'model_type', or 'features' in request")
            return jsonify({"error": "Missing 'dataset', 'model_type', or 'features' in request"}), 400

        dataset = data["dataset"]
        model_type = data["model_type"]
        features = data["features"]

        # Validate dataset and model_type
        if dataset not in models:
            logger.error(f"Invalid dataset: {dataset}")
            return jsonify({"error": f"Invalid dataset: {dataset}"}), 400
        if model_type not in models[dataset]:
            logger.error(f"Invalid model_type: {model_type}")
            return jsonify({"error": f"Invalid model_type: {model_type}"}), 400

        model = models[dataset][model_type]

        # Convert features to a DataFrame with proper column order
        expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(features.keys())
        input_data = pd.DataFrame([features], columns=expected_features)

        # Handle missing features
        missing_cols = set(expected_features) - set(features.keys())
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}")
            return jsonify({"error": f"Missing features: {missing_cols}"}), 400

        # Predict using the model
        prediction = model.predict(input_data)[0]

        # If model supports predict_proba, get fraud probability
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]  # Probability of fraud (class 1)

        # Convert prediction to a human-readable label
        prediction_label = "fraud" if prediction == 1 else "not fraud"

        logger.info(f"Prediction for dataset={dataset}, model_type={model_type}: {prediction_label}")

        response = {"prediction": prediction_label}
        if prob is not None:
            response["fraud_probability"] = round(prob, 4)

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
