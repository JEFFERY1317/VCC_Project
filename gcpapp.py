# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import time
import os
import psutil

app = Flask(__name__)

# Load the trained model
model = joblib.load('housing_model.joblib')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    data = request.get_json(force=True)
    input_data = pd.DataFrame([data])
    prediction = model.predict(input_data)[0]

    # Gather system metrics
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 / 1024  # in MB
    cpu_use = psutil.cpu_percent()

    response_time = time.time() - start_time
    return jsonify({
        'prediction': float(prediction),
        'response_time_ms': response_time * 1000,
        'memory_usage_mb': memory_use,
        'cpu_usage_percent': cpu_use
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    start_time = time.time()
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data)
    predictions = model.predict(input_data).tolist()

    # System metrics
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 / 1024
    cpu_use = psutil.cpu_percent()

    response_time = time.time() - start_time
    return jsonify({
        'predictions': predictions,
        'response_time_ms': response_time * 1000,
        'memory_usage_mb': memory_use,
        'cpu_usage_percent': cpu_use,
        'batch_size': len(data)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
