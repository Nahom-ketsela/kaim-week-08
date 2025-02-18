from flask import Flask, request, jsonify
import logging
import joblib
import pandas as pd
import os
import sys


# Initialize Flask

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Load Your Models

MODEL_DIR = "models"  # directory for pkl files

# We'll assume you have 2 "datasets": "fraud" & "credit"
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

# Load each model
for dataset, model_files in model_filenames.items():
    for model_type, filename in model_files.items():
        try:
            model_path = os.path.join(MODEL_DIR, filename)
            loaded_model = joblib.load(model_path)
            models[dataset][model_type] = loaded_model
            logger.info(f"✅ Loaded model '{model_type}' for dataset '{dataset}'")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"❌ Could not load model '{model_type}' ({dataset}): {e}")


# Prediction Endpoint

@app.route('/predict', methods=['POST'])
def predict():
    """
    POST JSON example:
    {
      "dataset": "fraud" or "credit",
      "model_type": "logistic_regression" or "decision_tree" etc.,
      "features": { "col1": val1, "col2": val2, ... }
    }
    """
    try:
        logger.info("🔵 Received prediction request")
        data = request.get_json()

        # Validate
        if not data or not {"dataset", "model_type", "features"}.issubset(data.keys()):
            return jsonify({"error": "Missing 'dataset', 'model_type', or 'features'"}), 400

        dataset = data["dataset"]
        model_type = data["model_type"]
        features = data["features"]

        # Check validity
        if dataset not in models or model_type not in models[dataset]:
            return jsonify({"error": "Invalid dataset or model_type"}), 400

        model = models[dataset][model_type]

        # If model was trained with feature_names_in_, use them; otherwise fallback
        if hasattr(model, "feature_names_in_"):
            expected_features = list(model.feature_names_in_)
        else:
            expected_features = list(features.keys())

        # Create a DataFrame from the input features
        input_df = pd.DataFrame([features])

        # Fill missing columns with 0
        missing_cols = set(expected_features) - set(input_df.columns)
        for col in missing_cols:
            input_df[col] = 0

        # Reorder columns to match the model
        input_df = input_df[expected_features]

        # Predict
        prediction = model.predict(input_df)[0]

        # Probability (if supported)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_df)[0][1]

        # Convert numeric prediction to label
        prediction_label = "fraud" if prediction == 1 else "not fraud"

        # Build response
        response = {"prediction": prediction_label}
        if prob is not None:
            response["fraud_probability"] = round(float(prob), 4)

        logger.info(f"✅ Prediction: {prediction_label} (prob={prob})")
        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500


#  Flask Endpoints for the CSV Data

DATA_FILE = "data/Cleaned_Fraud_Data.csv"

@app.route('/api/fraud_data', methods=['GET'])
def fraud_data():
    """
    Returns the entire CSV as JSON (list of rows as dicts).
    We'll rename columns to match what the dashboard code expects:
      - class -> is_fraud
      - purchase_time -> transaction_time
      - device_id -> device_type
    """
    try:
        df = pd.read_csv(DATA_FILE)
        
        # Rename columns so the dashboard references remain consistent
        df.rename(columns={
            "class": "is_fraud",
            "purchase_time": "transaction_time",
            "device_id": "device_type"
        }, inplace=True)
        
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        logger.error(f"❌ Error reading data file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/fraud_summary', methods=['GET'])
def fraud_summary():
    """
    Returns summary stats:
      - total_transactions
      - total_fraud
      - fraud_percentage
    """
    try:
        df = pd.read_csv(DATA_FILE)
        # Rename 'class' -> 'is_fraud' for consistency
        df.rename(columns={"class": "is_fraud"}, inplace=True)

        total_tx = len(df)
        total_fraud = df['is_fraud'].sum()  # 1 means fraud, 0 means not fraud
        fraud_pct = (total_fraud / total_tx * 100) if total_tx else 0.0

        return jsonify({
            "total_transactions": int(total_tx),
            "total_fraud": int(total_fraud),
            "fraud_percentage": round(fraud_pct, 2)
        })
    except Exception as e:
        logger.error(f"❌ Error reading data file: {e}")
        return jsonify({"error": str(e)}), 500

# Dash Dashboard Setup

import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import requests

dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

dash_app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    
    # Summary boxes
    html.Div(id="summary-stats", style={'display': 'flex', 'gap': '40px'}),
    
    # Fraud Over Time
    html.Div([
        html.H2("Fraud Cases Over Time"),
        dcc.Graph(id="line-chart")
    ]),
    
    # Fraud by Device/Browser
    html.Div([
        html.H2("Fraud by Device and Browser"),
        dcc.Graph(id="bar-chart")
    ])
])

@dash_app.callback(
    Output("summary-stats", "children"),
    Input("summary-stats", "id")
)
def update_summary(_):
    """
    Hit the /api/fraud_summary endpoint to get overall stats,
    then render them as boxes.
    """
    try:
        resp = requests.get("http://127.0.0.1:5000/api/fraud_summary")
        data = resp.json()
        
        box_style = {
            'border': '1px solid #ccc',
            'padding': '20px',
            'width': '200px',
            'textAlign': 'center'
        }
        return [
            html.Div([
                html.H3("Total Transactions"),
                html.H1(data['total_transactions'])
            ], style=box_style),
            html.Div([
                html.H3("Total Fraud"),
                html.H1(data['total_fraud'])
            ], style=box_style),
            html.Div([
                html.H3("Fraud %"),
                html.H1(f"{data['fraud_percentage']}%")
            ], style=box_style)
        ]
    except Exception as e:
        return [html.Div(f"Error: {e}")]

@dash_app.callback(
    Output("line-chart", "figure"),
    Input("line-chart", "id")
)
def update_line_chart(_):
    """
    Pull data from /api/fraud_data, group by day, and show
    daily fraud counts over time (line chart).
    """
    try:
        resp = requests.get("http://127.0.0.1:5000/api/fraud_data")
        df = pd.DataFrame(resp.json())

        # Convert 'transaction_time' to datetime
        df['transaction_time'] = pd.to_datetime(df['transaction_time'])

        # We assume 1 = fraud, 0 = not fraud
        fraud_df = df[df['is_fraud'] == 1]

        # Group by day
        daily = fraud_df.resample('D', on='transaction_time').size().reset_index(name='count')

        fig = px.line(daily, x='transaction_time', y='count', title="Fraud Cases Over Time")
        fig.update_layout(xaxis_title="Date", yaxis_title="Fraud Count")
        return fig
    except Exception as e:
        return px.line(title=f"Error: {e}")

@dash_app.callback(
    Output("bar-chart", "figure"),
    Input("bar-chart", "id")
)
def update_bar_chart(_):
    """
    Pull data from /api/fraud_data, group by device_type/browser,
    and show number of fraud cases per combination.
    """
    try:
        resp = requests.get("http://127.0.0.1:5000/api/fraud_data")
        df = pd.DataFrame(resp.json())

        # Filter only rows where is_fraud = 1
        fraud_only = df[df['is_fraud'] == 1]

        # Group by device_type, browser
        group_data = (fraud_only
                      .groupby(["device_type", "browser"])
                      .size()
                      .reset_index(name="fraud_count"))

        fig = px.bar(
            group_data,
            x="device_type",
            y="fraud_count",
            color="browser",
            barmode="group",
            title="Fraud by Device and Browser"
        )
        return fig
    except Exception as e:
        return px.bar(title=f"Error: {e}")

# -----------------------------------------------------------
# 6) Run the Combined App
# -----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
