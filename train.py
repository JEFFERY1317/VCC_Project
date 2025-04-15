from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib
import os
import argparse
import json
import matplotlib.pyplot as plt
from azureml.core import Run

# Get the run context
run = Run.get_context()

# Add argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=10)
parser.add_argument('--min_samples_split', type=int, default=2)
parser.add_argument('--min_samples_leaf', type=int, default=1)
parser.add_argument('--output-dir', type=str, default='./outputs')
args = parser.parse_args()

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load data
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Log dataset properties
run.log("dataset_size", X.shape[0])
run.log("num_features", X.shape[1])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Log hyperparameters
run.log("n_estimators", args.n_estimators)
run.log("max_depth", args.max_depth)
run.log("min_samples_split", args.min_samples_split)
run.log("min_samples_leaf", args.min_samples_leaf)

# Start timer
import time
start_time = time.time()

# Train model
model = RandomForestRegressor(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth if args.max_depth > 0 else None,
    min_samples_split=args.min_samples_split,
    min_samples_leaf=args.min_samples_leaf,
    random_state=42
)

model.fit(X_train, y_train)

# Log training time
training_time = time.time() - start_time
run.log("training_time", training_time)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Log metrics
run.log("mse", mse)
run.log("r2", r2)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Training Time: {training_time:.2f} seconds")

# Create a scatter plot of predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Housing Prices')
plt.savefig(os.path.join(args.output_dir, 'prediction_plot.png'))

# Log the plot
run.log_image("prediction_plot", os.path.join(args.output_dir, 'prediction_plot.png'))

# Save feature importances plot
feature_importance = pd.DataFrame(
    model.feature_importances_,
    index=X.columns,
    columns=['importance']
).sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', figsize=(10, 6))
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig(os.path.join(args.output_dir, 'feature_importance.png'))

# Log the feature importance plot
run.log_image("feature_importance", os.path.join(args.output_dir, 'feature_importance.png'))

# Save model and feature names
joblib.dump(model, os.path.join(args.output_dir, 'housing_model.joblib'))
with open(os.path.join(args.output_dir, 'feature_names.json'), 'w') as f:
    json.dump(list(X.columns), f)

print("Training completed successfully.")
