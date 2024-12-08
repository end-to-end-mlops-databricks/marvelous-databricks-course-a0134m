"""
This script evaluates and compares a new house price prediction model against the currently deployed model.
Key functionality:
- Loads test data and performs feature engineering
- Generates predictions using both new and existing models
- Calculates and compares performance metrics (MAE and RMSE)
- Registers the new model if it performs better
- Sets task values for downstream pipeline steps

The evaluation process:
1. Loads models from the serving endpoint
2. Prepares test data with feature engineering
3. Generates predictions from both models
4. Calculates error metrics
5. Makes registration decision based on MAE comparison
6. Updates pipeline task values with results
"""

import argparse

import mlflow
from databricks import feature_engineering
from databricks.sdk import WorkspaceClient
from pyspark.dbutils import DBUtils
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit, when
from workspace_utils import check_if_endpoint_exists, get_env_config_file, get_model_entity_index

from wine_quality.config import ProjectConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)
parser.add_argument(
    "--new_model_uri",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--job_run_id",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--git_sha",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
new_model_uri = args.new_model_uri
job_run_id = args.job_run_id
git_sha = args.git_sha

# config_path = (f"{root_path}/project_config.yml")
# config_path = ("/Volumes/mlops_test/house_prices/data/project_config.yml")
config_file_name = get_env_config_file(args.env)
config_path = f"{root_path}/{config_file_name}"
config = ProjectConfig.from_yaml(config_path=config_path)

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name
endpoint_name = "wine-quality-model-serving-fe"
model_name = "wine-quality-model-fe"
full_model_name = f"{catalog_name}.{schema_name}.{model_name}"


if check_if_endpoint_exists(workspace, endpoint_name):
    # Define the serving endpoint
    serving_endpoint = workspace.serving_endpoints.get(endpoint_name)

    # This assumes that only one model is served by the endpoint
    # model_name = serving_endpoint.config.served_models[0].model_name
    # model_version = serving_endpoint.config.served_models[0].model_version
    # print(model_name)
    # print(model_version)

    # Loop over served entities and get model name and version. Can be used for multiple models.
    # served_entity_version = model_entity_version(workspace, endpoint_name, f"{catalog_name}.{schema_name}.{model_name}")
    served_model_index = get_model_entity_index(workspace, endpoint_name, full_model_name)

    if served_model_index < 0:
        raise ValueError("Model not found in the serving endpoint.")

    model_name = serving_endpoint.config.served_models[served_model_index].model_name
    model_version = serving_endpoint.config.served_models[served_model_index].model_version
    print(model_name)
    print(model_version)

    previous_model_uri = f"models:/{model_name}/{model_version}"

    # Load test set and create additional features in Spark DataFrame
    test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")
    # Cast residual_sugar to int for the function inputs
    test_set = test_set.withColumn("residual_sugar", test_set["residual_sugar"].cast("int"))
    test_set = test_set.withColumn("is_sweet_indicator", when(col("residual_sugar") > 40, lit("1")).otherwise(lit("0")))
    test_set = test_set.withColumn("is_sweet_indicator", test_set["residual_sugar"].cast("int"))
    test_set = test_set.withColumn("Id", test_set["id"].cast("string"))

    # Select the necessary columns for prediction and target
    X_test_spark = test_set.select(num_features + ["is_sweet_indicator", "Id"])
    y_test_spark = test_set.select("Id", target)

    # Generate predictions from both models
    predictions_previous = fe.score_batch(model_uri=previous_model_uri, df=X_test_spark)
    predictions_new = fe.score_batch(model_uri=new_model_uri, df=X_test_spark)

    predictions_new = predictions_new.withColumnRenamed("prediction", "prediction_new")
    predictions_old = predictions_previous.withColumnRenamed("prediction", "prediction_old")
    test_set = test_set.select("Id", "quality")

    # Join the DataFrames on the 'id' column
    df = test_set.join(predictions_new, on="Id").join(predictions_old, on="Id")

    # Calculate the absolute error for each model
    df = df.withColumn("error_new", F.abs(df["quality"] - df["prediction_new"]))
    df = df.withColumn("error_old", F.abs(df["quality"] - df["prediction_old"]))

    # Calculate the absolute error for each model
    df = df.withColumn("error_new", F.abs(df["quality"] - df["prediction_new"]))
    df = df.withColumn("error_old", F.abs(df["quality"] - df["prediction_old"]))

    # Calculate the Mean Absolute Error (MAE) for each model
    mae_new = df.agg(F.mean("error_new")).collect()[0][0]
    mae_old = df.agg(F.mean("error_old")).collect()[0][0]

    # Calculate the Root Mean Squared Error (RMSE) for each model
    evaluator = RegressionEvaluator(labelCol="quality", predictionCol="prediction_new", metricName="rmse")
    rmse_new = evaluator.evaluate(df)

    evaluator.setPredictionCol("prediction_old")
    rmse_old = evaluator.evaluate(df)

    # Compare models based on MAE and RMSE
    print(f"MAE for New Model: {mae_new}")
    print(f"MAE for Old Model: {mae_old}")

    if mae_new < mae_old:
        print("New model is better based on MAE.")
        model_version = mlflow.register_model(
            model_uri=new_model_uri, name=full_model_name, tags={"git_sha": f"{git_sha}", "job_run_id": job_run_id}
        )

        print("New model registered with version:", model_version.version)
        dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
        dbutils.jobs.taskValues.set(key="model_update", value=1)
    else:
        print("Old model is better based on MAE.")
        dbutils.jobs.taskValues.set(key="model_update", value=0)
else:
    print("Endpoint not found, registering new model.")
    model_version = mlflow.register_model(
        model_uri=new_model_uri, name=full_model_name, tags={"git_sha": f"{git_sha}", "job_run_id": job_run_id}
    )

    print("New model registered with version:", model_version.version)
    dbutils.jobs.taskValues.set(key="model_version", value=model_version.version)
    dbutils.jobs.taskValues.set(key="model_update", value=1)
