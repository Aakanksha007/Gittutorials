
import os
import mlflow
 
# Set the tracking URI to your MLflow server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000'
 
# Create a new experiment
experiment_name = "my_experiment_1"
experiment_id = mlflow.create_experiment(experiment_name)
 
print(f"Experiment created with ID: {experiment_id}")
 
 