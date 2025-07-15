# Serialization & Formats
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

X_train, y_train, X_test = None, None, None

model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# Cloudpickle format (default) - better cross-system compatibility
mlflow.sklearn.log_model(
    sk_model=model,
    name="cloudpickle_model",
    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
)

# Pickle format - faster but less portable
mlflow.sklearn.log_model(
    sk_model=model,
    name="pickle_model",
    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
)

# Model Signatures
from mlflow.models import infer_signature
import pandas as pd

# Create model signature automatically
X_sample = X_train[:100]
predictions = model.predict(X_sample)
signature = infer_signature(X_sample, predictions)

# Log model with signature for production safety
mlflow.sklearn.log_model(
    sk_model=model,
    name="model_with_signature",
    signature=signature,
    input_example=X_sample[:5],  # Include example for documentation
)


# Loading & Usage
# Load model in different ways
import mlflow.sklearn
import mlflow.pyfunc

run_id = "your_run_id_here"

# Load as scikit-learn model (preserves all sklearn functionality)
sklearn_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
predictions = sklearn_model.predict(X_test)

# Load as PyFunc model (generic Python function interface)
pyfunc_model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
predictions = pyfunc_model.predict(pd.DataFrame(X_test))

# Load from model registry (production deployment)
registered_model = mlflow.pyfunc.load_model("models:/MyModel@champion")
