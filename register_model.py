from azureml.core import Workspace, Experiment, Model
import os

# Get workspace
subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "<your-subscription-id>")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "house-rg")
workspace_name = os.environ.get("AZURE_WORKSPACE_NAME", "housing-ml-workspace")

ws = Workspace.get(
    name=workspace_name,
    subscription_id=subscription_id,
    resource_group=resource_group
)

# Get experiment
experiment_name = 'housing-price-prediction'
experiment = Experiment(workspace=ws, name=experiment_name)

# Get the latest run
runs = list(experiment.get_runs())
latest_run = sorted(runs, key=lambda r: r.get_details()['startTimeUtc'], reverse=True)[0]

print(f"Latest run ID: {latest_run.id}")
print(f"Latest run status: {latest_run.get_status()}")

# Register model from the latest run
model = latest_run.register_model(
    model_name='housing-price-model',
    model_path='outputs/housing_model.joblib',
    description='Random Forest model for housing price prediction',
    tags={
        'framework': 'scikit-learn', 
        'algorithm': 'RandomForest',
        'mse': round(latest_run.get_metrics()['mse'], 4),
        'r2': round(latest_run.get_metrics()['r2'], 4)
    }
)

print(f"Model registered: {model.name}, version: {model.version}")
