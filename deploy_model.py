from azureml.core import Workspace, Model
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig
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

# Get registered model
model_name = 'housing-price-model'
model = Model(ws, model_name)
print(f"Model: {model.name}, Version: {model.version}")

# Create inference config
env = ws.environments['housing-prediction-env']
inference_config = InferenceConfig(
    entry_script="score.py",
    environment=env
)

# Set deployment configuration
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    auth_enabled=True,
    enable_app_insights=True,
    description="Housing price prediction service"
)

# Deploy model
service_name = 'housing-price-service'
service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(f"Service deployment status: {service.state}")

# Print service details
print(f"Service URL: {service.scoring_uri}")
