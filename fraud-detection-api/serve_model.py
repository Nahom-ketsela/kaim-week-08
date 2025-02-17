from flask import Flask, request, jsonify
import logging
import joblib
import pandas as pd
import os
import sys

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

# Load models safely and log expected feature names
for dataset, model_files in model_filenames.items():
    for model_type, filename in model_files.items():
        try:
            model_path = os.path.join(MODEL_DIR, filename)
            models[dataset][model_type] = joblib.load(model_path)
            logger.info(f"✅ Loaded model: {model_type} for dataset: {dataset}")

            # Log expected feature names
            model = models[dataset][model_type]
            if hasattr(model, "feature_names_in_"):
                feature_names = model.feature_names_in_
                logger.info(f"📌 Features expected by {model_type} ({dataset}): {feature_names}")
            else:
                logger.warning(f"⚠️ Model {model_type} ({dataset}) does not have 'feature_names_in_'. It might have been trained without named columns.")

            # Flush logs to ensure they print immediately
            sys.stdout.flush()

        except Exception as e:
            logger.error(f"❌ Error loading model {model_type} ({dataset}): {e}")

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
        logger.info("🔵 Received prediction request")

        # Get JSON data from the request
        data = request.get_json()

        # Validate required fields
        required_fields = {"dataset", "model_type", "features"}
        if not data or not required_fields.issubset(data.keys()):
            logger.error("❌ Missing 'dataset', 'model_type', or 'features' in request")
            return jsonify({"error": "Missing 'dataset', 'model_type', or 'features'"}), 400

        dataset = data["dataset"]
        model_type = data["model_type"]
        features = data["features"]

        # Validate dataset and model_type
        if dataset not in models:
            logger.error(f"❌ Invalid dataset: {dataset}")
            return jsonify({"error": f"Invalid dataset: {dataset}"}), 400
        if model_type not in models[dataset]:
            logger.error(f"❌ Invalid model_type: {model_type}")
            return jsonify({"error": f"Invalid model_type: {model_type}"}), 400

        model = models[dataset][model_type]

        # Ensure input has the correct feature order
        expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else list(features.keys())

        # Create DataFrame with missing features handled
        input_data = pd.DataFrame([features])
        missing_cols = set(expected_features) - set(input_data.columns)

        if missing_cols:
            logger.warning(f"⚠️ Missing features: {missing_cols} (filled with 0)")
            for col in missing_cols:
                input_data[col] = 0

        input_data = input_data[expected_features]  # Reorder columns correctly

        # Predict using the model
        prediction = model.predict(input_data)[0]

        # Get fraud probability if model supports predict_proba
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_data)[0][1]  # Probability of fraud (class 1)

        # Convert prediction to a readable label
        prediction_label = "fraud" if prediction == 1 else "not fraud"

        logger.info(f"✅ Prediction for dataset={dataset}, model_type={model_type}: {prediction_label}")

        response = {"prediction": prediction_label}
        if prob is not None:
            response["fraud_probability"] = round(prob, 4)

        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
