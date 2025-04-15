from azureml.core import Workspace
from azureml.core import Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import os

# Create workspace
subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID", "<your-subscription-id>")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP", "house-rg")
workspace_name = os.environ.get("AZURE_WORKSPACE_NAME", "housing-ml-workspace")

try:
    ws = Workspace.get(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group
    )
    print("Found existing workspace.")
except Exception:
    print("Creating new workspace...")
    ws = Workspace.create(
        name=workspace_name,
        subscription_id=subscription_id,
        resource_group=resource_group,
        location="centralindia",  # Replace with your preferred region
        create_resource_group=True,
        exist_ok=True
    )

# Create compute cluster
compute_name = "housing-compute"

try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print("Found existing compute target.")
except ComputeTargetException:
    print("Creating new compute target...")
    
    compute_config = AmlCompute.provisioning_configuration(
        vm_size="STANDARD_DS3_V2",
        min_nodes=0,
        max_nodes=2,
        idle_seconds_before_scaledown=1800
    )
    
    compute_target = ComputeTarget.create(ws, compute_name, compute_config)
    compute_target.wait_for_completion(show_output=True)

# Print workspace info
print(f"Workspace: {ws.name}")
print(f"Resource Group: {ws.resource_group}")
print(f"Location: {ws.location}")
print(f"Compute Target: {compute_target.name}")
