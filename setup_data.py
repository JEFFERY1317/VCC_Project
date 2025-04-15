import pandas as pd
from sklearn.datasets import fetch_california_housing
from sqlalchemy import create_engine
import time
import os

time.sleep(10)

db_user = os.environ.get('DB_USER', 'mluser')
db_password = os.environ.get('DB_PASSWORD', 'password')
db_host = os.environ.get('DB_HOST', 'database')
db_name = os.environ.get('DB_NAME', 'housing_data')

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

print(f"Dataset shape: {df.shape}")
print(df.head())
print(df.describe())

connection_string = f'mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}'
print(f"Connecting to database: {connection_string}")

engine = create_engine(connection_string)
df.to_sql('housing', engine, if_exists='replace', index=False)
print("Data successfully loaded into MySQL database")
