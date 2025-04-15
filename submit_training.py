from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.runconfig import RunConfiguration
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

# Get environment
env = Environment.get(workspace=ws, name="housing-prediction-env")

# Get compute target
compute_name = "housing-compute"
compute_target = ws.compute_targets[compute_name]

# Create run configuration
run_config = RunConfiguration()
run_config.environment = env
run_config.target = compute_target

# Create ScriptRunConfig
src = ScriptRunConfig(
    source_directory='.',
    script='train.py',
    arguments=[
        '--n_estimators', 100,
        '--max_depth', 10,
        '--min_samples_split', 2,
        '--min_samples_leaf', 1
    ],
    compute_target=compute_target,
    environment=env
)

# Create experiment
experiment_name = 'housing-price-prediction'
experiment = Experiment(workspace=ws, name=experiment_name)

# Submit experiment
run = experiment.submit(src)

# Print run details
print(f"Run ID: {run.id}")
print(f"Run portal URL: {run.get_portal_url()}")

# Wait for completion
print("Waiting for run completion...")
run.wait_for_completion(show_output=True)

# Print metrics
print("\nRun Metrics:")
print(f"MSE: {run.get_metrics()['mse']:.4f}")
print(f"R² Score: {run.get_metrics()['r2']:.4f}")
print(f"Training Time: {run.get_metrics()['training_time']:.2f} seconds")
