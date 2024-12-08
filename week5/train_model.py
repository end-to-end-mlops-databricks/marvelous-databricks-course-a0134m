"""
This script trains a LightGBM model for house price prediction with feature engineering.
Key functionality:
- Loads training and test data from Databricks tables
- Performs feature engineering using Databricks Feature Store
- Creates a pipeline with preprocessing and LightGBM regressor
- Tracks the experiment using MLflow
- Logs model metrics, parameters and artifacts
- Handles feature lookups and custom feature functions
- Outputs model URI for downstream tasks

The model uses numerical features, including a custom calculated wine sweetness feature.
"""

import argparse

import mlflow
from databricks import feature_engineering
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from databricks.sdk import WorkspaceClient
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from workspace_utils import get_env_config_file

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
    "--git_sha",
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
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path
git_sha = args.git_sha
job_run_id = args.job_run_id
cur_experiment_name = "/Shared/wine-quality-fe"
cur_model_artifact_path = "wine-quality-model-fe"


# config_path = (f"{root_path}/project_config.yml")
# config_path = ("/Volumes/mlops_students/mahajan134/mlops_vol/project_config.yml")
config_file_name = get_env_config_file(args.env)
config_path = f"{root_path}/{config_file_name}"
config = ProjectConfig.from_yaml(config_path=config_path)

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

# Extract configuration details
num_features = config.num_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name


# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.wine_features"
function_name = f"{catalog_name}.{schema_name}.calculate_wine_sweetness"

# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").drop("volatile_acidity", "alcohol", "sulphates")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

# Cast residual_sugar to int for the function input
train_set = train_set.withColumn("residual_sugar", train_set["residual_sugar"].cast("int"))
train_set = train_set.withColumn("Id", train_set["id"].cast("string"))


# Feature engineering setup
training_set = fe.create_training_set(
    df=train_set,
    label=target,
    feature_lookups=[
        FeatureLookup(
            table_name=feature_table_name,
            feature_names=["volatile_acidity", "alcohol", "sulphates"],
            lookup_key="Id",
        ),
        FeatureFunction(
            udf_name=function_name,
            output_name="is_sweet_indicator",
            input_bindings={"residual_sugar_content": "residual_sugar"},
        ),
    ],
    exclude_columns=["update_timestamp_utc"],
)

# Load feature-engineered DataFrame
training_df = training_set.load_df().toPandas()

# Calculate is_sweet_indicator for training and test set
test_set["is_sweet_indicator"] = test_set["residual_sugar"].apply(lambda x: 1 if x > 40 else 0)

# Split features and target
X_train = training_df[num_features + ["is_sweet_indicator"]]
y_train = training_df[target]
X_test = test_set[num_features + ["is_sweet_indicator"]]
y_test = test_set[target]

# Setup preprocessing and model pipeline
preprocessor = ColumnTransformer(
    transformers=[("num", SimpleImputer(strategy="mean"), num_features), ("std", StandardScaler(), num_features)],
    remainder="passthrough",
)

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", LGBMRegressor(**parameters))])


mlflow.set_experiment(experiment_name=cur_experiment_name)

with mlflow.start_run(tags={"branch": "week5", "git_sha": f"{git_sha}", "job_run_id": job_run_id}) as run:
    run_id = run.info.run_id
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate and print metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log model parameters, metrics, and model
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    # Log model with feature engineering
    fe.log_model(
        model=pipeline,
        flavor=mlflow.sklearn,
        artifact_path=cur_model_artifact_path,
        training_set=training_set,
        signature=signature,
    )

model_uri = f"runs:/{run_id}/{cur_model_artifact_path}"
dbutils.jobs.taskValues.set(key="new_model_uri", value=model_uri)
