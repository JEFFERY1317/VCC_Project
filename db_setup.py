import pandas as pd
from sklearn.datasets import fetch_california_housing
from sqlalchemy import create_engine

# Load California Housing dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target

# Replace credentials/IP
db_connection = "mysql+pymysql://projectuser:Password@34.68.197.208/projectdata"
engine = create_engine(db_connection)

df.to_sql('housing', engine, if_exists='replace', index=False)
print(f"Loaded {len(df)} rows to database")
