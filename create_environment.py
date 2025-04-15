from azureml.core import Workspace, Environment
from azureml.core.conda_dependencies import CondaDependencies
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

# Create environment
env = Environment(name="housing-prediction-env")
conda_dep = CondaDependencies()

# Adding packages
conda_dep.add_conda_package("python=3.8")
conda_dep.add_conda_package("scikit-learn=0.24.2")
conda_dep.add_conda_package("pandas=1.3.0")
conda_dep.add_conda_package("numpy=1.20.3")
conda_dep.add_pip_package("azureml-defaults")
conda_dep.add_pip_package("matplotlib")
conda_dep.add_pip_package("joblib")
conda_dep.add_pip_package("pytest")

env.python.conda_dependencies = conda_dep

# Register environment
env.register(workspace=ws)
print(f"Environment '{env.name}' registered.")
