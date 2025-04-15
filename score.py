import json
import numpy as np
import os
import pandas as pd
import joblib
import time

def init():
    global model, feature_names
    
    # Get model path and load model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'housing_model.joblib')
    model = joblib.load(model_path)
    
    # Load feature names
    feature_names_path = os.path.join(os.path.dirname(model_path), 'feature_names.json')
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            feature_names = json.load(f)
    else:
        feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    
    print("Model loaded successfully")

def run(input_data):
    start_time = time.time()
    
    try:
        # Parse input and make prediction
        data = json.loads(input_data)
        df = pd.DataFrame(data['data'])
        
        # Ensure all required features are present
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0.0
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Make predictions
        predictions = model.predict(df).tolist()
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Return predictions with response time
        return json.dumps({
            'predictions': predictions,
            'response_time_ms': response_time * 1000
        })
    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
