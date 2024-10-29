# Databricks notebook source

from pyspark.sql import SparkSession
from wine_quality.config import ProjectConfig
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri('databricks-uc') # It must be -uc for registering models to Unity Catalog
cur_experiment_name = "/Shared/wine-quality-basic"
git_sha_id = "a14d4efd57456e49657cd2b6c40518b90c28407f"

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details
num_features = config.num_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# COMMAND ----------
spark = SparkSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set[num_features]
y_train = train_set[target]

X_test = test_set[num_features]
y_test = test_set[target]

# COMMAND ----------
# Define the preprocessor and do some preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), num_features),
        ('std', StandardScaler(), num_features)
    ], remainder="passthrough"
)

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[
     ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(**parameters))
])

# COMMAND ----------
mlflow.set_experiment(experiment_name=cur_experiment_name)
# Specify model type during auto logging because pipeline is not a scikit-learn but model is LightGBM
mlflow.lightgbm.autolog()

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": f"{git_sha_id}",
          "branch": "week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)

    # Create signature object, required for registering the model in UC. Signature is the expected schema for inputs and outputs.
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="lightgbm-pipeline-model",
        signature=signature
    )

# COMMAND ----------
model_version = mlflow.register_model(
    model_uri=f'runs:/{run_id}/lightgbm-pipeline-model',
    name=f"{catalog_name}.{schema_name}.wine_quality_basic",
    tags={"git_sha": f"{git_sha_id}"})

# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)

# COMMAND ----------
dataset_source.load() 