# Model Registry

# Register model to MLflow Model Registry
import mlflow
from mlflow import MlflowClient

client = MlflowClient()

# Log and register model in one step
with mlflow.start_run():
    mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        registered_model_name="CustomerChurnModel",
        signature=signature,
    )

# Or register an existing model
run_id = "your_run_id"
model_uri = f"runs:/{run_id}/model"

# Register the model
registered_model = mlflow.register_model(model_uri=model_uri, name="CustomerChurnModel")

# Use aliases instead of deprecated stages for deployment management
# Set aliases for different deployment environments
client.set_registered_model_alias(
    name="CustomerChurnModel",
    alias="champion",  # Production model
    version=registered_model.version,
)

client.set_registered_model_alias(
    name="CustomerChurnModel",
    alias="challenger",  # A/B testing model
    version=registered_model.version,
)

# Use tags to track model status and metadata
client.set_model_version_tag(
    name="CustomerChurnModel",
    version=registered_model.version,
    key="validation_status",
    value="approved",
)

client.set_model_version_tag(
    name="CustomerChurnModel",
    version=registered_model.version,
    key="deployment_date",
    value="2025-05-29",
)


# Model Serving

# Serve model using alias for production deployment
# mlflow models serve \
#     -m "models:/CustomerChurnModel@champion" \
#     -p 5002 \
#     --no-conda

import requests
import json

# Example prediction request
data = {"inputs": [[1.2, 0.8, 3.4, 2.1]]}  # Feature values

response = requests.post(
    "http://localhost:5002/invocations",
    headers={"Content-Type": "application/json"},
    data=json.dumps(data),
)

predictions = response.json()
