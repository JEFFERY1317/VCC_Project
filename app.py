from flask import Flask, request, jsonify
import pandas as pd
import joblib
import time
import os
import psutil
import json
import socket

app = Flask(__name__)

# Get hostname for identifying container
HOSTNAME = socket.gethostname()

# Load the model
model = joblib.load('housing_model.joblib')

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "container": HOSTNAME
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Start timer
    start_time = time.time()
    
    # Get data from request
    data = request.get_json(force=True)
    
    # Convert to DataFrame
    input_data = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Get system metrics
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 / 1024  # in MB
    cpu_use = psutil.cpu_percent(interval=0.1)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Return prediction and metrics
    return jsonify({
        'prediction': float(prediction),
        'response_time_ms': response_time * 1000,
        'memory_usage_mb': memory_use,
        'cpu_usage_percent': cpu_use,
        'container_id': HOSTNAME
    })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    # Start timer
    start_time = time.time()
    
    # Get data from request
    data = request.get_json(force=True)
    
    # Convert to DataFrame
    input_data = pd.DataFrame(data)
    
    # Make prediction
    predictions = model.predict(input_data).tolist()
    
    # Get system metrics
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 / 1024  # in MB
    cpu_use = psutil.cpu_percent(interval=0.1)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Return predictions and metrics
    return jsonify({
        'predictions': predictions,
        'response_time_ms': response_time * 1000,
        'memory_usage_mb': memory_use,
        'cpu_usage_percent': cpu_use,
        'batch_size': len(data),
        'container_id': HOSTNAME
    })

@app.route('/metrics', methods=['GET'])
def metrics():
    process = psutil.Process(os.getpid())
    memory_use = process.memory_info().rss / 1024 / 1024  # in MB
    cpu_use = psutil.cpu_percent(interval=0.1)
    
    return jsonify({
        'memory_usage_mb': memory_use,
        'cpu_usage_percent': cpu_use,
        'container_id': HOSTNAME
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
