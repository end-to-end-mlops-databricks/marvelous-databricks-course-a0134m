# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from mlflow.models import infer_signature
from wine_quality.data_processor import ProjectConfig
import json
from mlflow import MlflowClient
from mlflow.utils.environment import _mlflow_conda_env
from wine_quality.utils import adjust_predictions

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")
client = MlflowClient()

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details
num_features = config.num_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

spark = SparkSession.builder.getOrCreate()

standard_experiment_name = "/Shared/wine-quality-basic" # Created in 03.log_and_register_model
custom_experiment_name = "/Shared/wine-quality-custom-pyfunc"
git_sha_id = "a14d4efd57456e49657cd2b6c40518b90c28407f"
custom_model_artifact_path = "custom-pyfunc-wine-quality-model"
custom_model_name = "wine-quality-model-custom-pyfunc"

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=[standard_experiment_name],
    filter_string="tags.branch='week2'",
).run_id[0]

# If we didn't use pipeline in the previous step, we will need to load mlflow.lightgbm.load_model
model = mlflow.sklearn.load_model(f'runs:/{run_id}/lightgbm-pipeline-model')

# COMMAND ----------

class WineQualityModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = {"Prediction": adjust_predictions(
                predictions[0])}
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")
        
# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features].toPandas()
y_test = test_set[[target]].toPandas()


# COMMAND ----------
wrapped_model = WineQualityModelWrapper(model) # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------
# this is a trick with custom packages
# https://docs.databricks.com/en/machine-learning/model-serving/private-libraries-model-serving.html
# but does not work with pyspark, so we have a better option :-)

mlflow.set_experiment(experiment_name=custom_experiment_name)
git_sha = git_sha_id

# Path for the whl file below in log_model can be volumes when running in a Databricks environment.
# Pip dependency path containing folder code is referring to model artificat path
with mlflow.start_run(tags={"branch": "week2",
                            "git_sha": f"{git_sha}"}) as run:
    
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={'Prediction': example_prediction})
    dataset = mlflow.data.from_spark(
        train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    conda_env = _mlflow_conda_env(
        additional_conda_deps=None,
        additional_pip_deps=["code/wine_quality-0.0.1-py3-none-any.whl",
                             ],
        additional_conda_channels=None,
    )
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path=custom_model_artifact_path,
        code_paths = ["../wine_quality-0.0.1-py3-none-any.whl"],
        signature=signature
    )

# COMMAND ----------
# We are loading pyfunc model here instead of sklearn model in 03.log_and_register_model.py

loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/{custom_model_artifact_path}')
loaded_model.unwrap_python_model()

# COMMAND ----------
model_name = f"{catalog_name}.{schema_name}.{custom_model_name}"

model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/{custom_model_artifact_path}',
    name=model_name,
    tags={"git_sha": f"{git_sha}"})

# COMMAND ----------
with open("model_version.json", "w") as json_file:
    json.dump(model_version.__dict__, json_file, indent=4)

# COMMAND ----------
model_version_alias = "the_best_model"
client.set_registered_model_alias(model_name, model_version_alias, "1")  

model_uri = f"models:/{model_name}@{model_version_alias}"
model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------
client.get_model_version_by_alias(model_name, model_version_alias)
# COMMAND ----------
model