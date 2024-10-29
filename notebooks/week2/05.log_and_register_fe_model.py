# Databricks notebook source
# MAGIC %pip install ../wine_quality-0.0.1-py3-none-any.whl

# COMMAND ----------

dbutils.library.restartPython() 

# COMMAND ----------
import yaml
from databricks import feature_engineering
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient
import mlflow
from pyspark.sql import functions as F
from lightgbm import LGBMRegressor
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from datetime import datetime
from databricks.feature_engineering import FeatureFunction, FeatureLookup
from wine_quality.config import ProjectConfig

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow.set_tracking_uri("databricks")

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Extract configuration details
num_features = config.num_features
target = config.target
parameters = config.parameters
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names and function name
feature_table_name = f"{catalog_name}.{schema_name}.wine_features"
function_name = f"{catalog_name}.{schema_name}.calculate_wine_quality"

# COMMAND ----------
# Load training and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# COMMAND ----------
# Create or replace the house_features table
# spark.sql(f"""
# CREATE OR REPLACE TABLE {catalog_name}.{schema_name}.wine_features
# (Id STRING NOT NULL,
#  OverallQual INT,
#  GrLivArea INT,
#  GarageCars INT);
# """)

# spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.wine_features "
#           "ADD CONSTRAINT wine_pk PRIMARY KEY(Id);")

# spark.sql(f"ALTER TABLE {catalog_name}.{schema_name}.wine_features "
#           "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")

# # Insert data into the feature table from both train and test sets
# spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.wine_features "
#           f"SELECT Id, OverallQual, GrLivArea, GarageCars FROM {catalog_name}.{schema_name}.train_set")
# spark.sql(f"INSERT INTO {catalog_name}.{schema_name}.wine_features "
#           f"SELECT Id, OverallQual, GrLivArea, GarageCars FROM {catalog_name}.{schema_name}.test_set")