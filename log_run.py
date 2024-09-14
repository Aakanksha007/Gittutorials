import os
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# Set the tracking URI to your MLflow server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'

# Load data
data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Set the experiment
experiment_name = "my_experiment_1"
mlflow.set_experiment(experiment_name)

# Start a new run
with mlflow.start_run():
    # Train a model
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Log the model
    mlflow.sklearn.log_model(model, "random_forest_model")

    # Log parameters and metrics
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("r2_score", model.score(X_test, y_test))

    print(f"Run ID: {mlflow.active_run().info.run_id}")
