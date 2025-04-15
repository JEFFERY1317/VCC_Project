import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import time
import os

# Get database connection parameters from environment variables
db_user = os.environ.get('DB_USER', 'mluser')
db_password = os.environ.get('DB_PASSWORD', 'password')
db_host = os.environ.get('DB_HOST', 'database')
db_name = os.environ.get('DB_NAME', 'housing_data')

# Connect to the database
connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
print(f"Connecting to database: {connection_string}")

engine = create_engine(connection_string)
df = pd.read_sql('SELECT * FROM housing', engine)

# Split features and target
X = df.drop('target', axis=1)
y = df['target']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
start_time = time.time()
print("Training Random Forest model...")

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

training_time = time.time() - start_time
print(f"Model trained in {training_time:.2f} seconds")

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Save the model
joblib.dump(model, 'housing_model.joblib')
print("Model saved as housing_model.joblib")

# Create a simple plot for visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Housing Prices')
plt.savefig('model_performance.png')
print("Performance visualization saved as model_performance.png")

# Also save model metrics to a file for the monitoring service
metrics = {
    'training_time': training_time,
    'mse': mse,
    'r2': r2
}

pd.DataFrame([metrics]).to_csv('model_metrics.csv', index=False)
print("Model metrics saved to model_metrics.csv")
