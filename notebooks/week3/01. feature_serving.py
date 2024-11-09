# Databricks notebook source
# MAGIC %pip install ../wine_quality-0.0.1-py3-none-any.whl

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------

"""
Create feature table in unity catalog, it will be a delta table
Create online table which uses the feature delta table created in the previous step
Create a feature spec. When you create a feature spec,
you specify the source Delta table.
This allows the feature spec to be used in both offline and online scenarios.
For online lookups, the serving endpoint automatically uses the online table to perform low-latency feature lookups.
The source Delta table and the online table must use the same primary key.

"""

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import pandas as pd
import requests
from databricks import feature_engineering
from databricks.feature_engineering import FeatureLookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy,
)
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.sql import SparkSession

from wine_quality.config import ProjectConfig

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy and query a feature serving endpoint
https://docs.databricks.com/en/machine-learning/feature-store/feature-serving-tutorial.html

# COMMAND ----------

spark = SparkSession.builder.getOrCreate()

# Initialize Databricks clients
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load config, train and test tables

# COMMAND ----------

# Load config
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# Get feature columns details
num_features = config.num_features
# cat_features = config.cat_features
target = config.target
catalog_name = config.catalog_name
schema_name = config.schema_name

# Define table names
feature_table_name = f"{catalog_name}.{schema_name}.wine_quality_preds"
online_table_name = f"{catalog_name}.{schema_name}.wine_quality_preds_online"

# Load training and test sets from Catalog
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

df = pd.concat([train_set, test_set])


# COMMAND ----------

# MAGIC %md
# MAGIC ## Load a registered model

# COMMAND ----------

# Load the MLflow model for predictions
pipeline = mlflow.sklearn.load_model(f"models:/{catalog_name}.{schema_name}.wine_quality_basic/3")

# COMMAND ----------

# Prepare the DataFrame for predictions and feature table creation - these features are the ones we want to serve.
preds_df = df[["id", "volatile_acidity", "alcohol", "sulphates"]]
preds_df["quality"] = pipeline.predict(df[num_features])

preds_df = spark.createDataFrame(preds_df)

# 1. Create the feature table in Databricks. Also referred sometimes as offline table.
# NOTE: Data type for id column must be STRING
fe.create_table(
    name=feature_table_name, primary_keys=["id"], df=preds_df, description="Wine quality predictions feature table"
)

# Enable Change Data Feed
spark.sql(f"""
    ALTER TABLE {feature_table_name}
    SET TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# COMMAND ----------

# 2. Create the online table using feature table created in the previous step

spec = OnlineTableSpec(
    primary_key_columns=["id"],
    source_table_full_name=feature_table_name,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy.from_dict({"triggered": "true"}),
    perform_full_copy=False,
)

#Create the online table in Databricks
try:
    online_table_pipeline = workspace.online_tables.create(name=online_table_name, spec=spec)
except Exception as e:
 if "already exists" in str(e):
   pass
 else:
   raise e

online_table_pipeline = workspace.online_tables.get(name=online_table_name)

# COMMAND ----------
# 3. Create feture look up and feature spec table feature table

# Define features to look up from the feature table
features = [
    FeatureLookup(
        table_name=feature_table_name, lookup_key="id", feature_names=[ "volatile_acidity", "alcohol", "sulphates", "quality"]
    )
]

# Create the feature spec  - This is entity that we serve in our model serving endpoint
feature_spec_name = f"{catalog_name}.{schema_name}.wine_quality_returned_predictions"

# Create the feature spec for serving
# The following code ignores errors raised if a feature_spec with the specified name already exists.
try:
    fe.create_feature_spec(name=feature_spec_name, features=features, exclude_columns=None)
except Exception as e:
    if "already exists" in str(e):
        pass
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy Feature Serving Endpoint

# COMMAND ----------
# 4. Create endpoing using feature spec


# Create a serving endpoint for the wine quality predictions
endpoint_name = "wine-quality-feature-serving"

try:
    status = workspace.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=feature_spec_name,  # feature spec name defined in the previous step
                    scale_to_zero_enabled=True, # Cost saving mechanism where the endpoint scales down to zero when not in use
                    workload_size="Small",  # Define the workload size (Small, Medium, Large)
                )
            ]
        ),
    )
    print(status)
except Exception as e:
    if "already exists" in str(e):
        pass
    else:
        raise e

# Get the status of the endpoint
status = workspace.serving_endpoints.get(name=endpoint_name)
print(status)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Call The Endpoint

# COMMAND ----------
# Option: Temp token from notebook context or use PAT - neither is best practice
# Production setup: In Azure  use M2M entra id which is valid for 16 min and you have to rotate it
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
host = spark.conf.get("spark.databricks.workspaceUrl")

# COMMAND ----------

id_list = preds_df["id"]

# COMMAND ----------
# See MLServe docs or MLFlow serving docs for more details on invocations pattern
start_time = time.time()
serving_endpoint = f"https://{host}/serving-endpoints/wine-quality-feature-serving/invocations"
response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_records": [{"id": "25"}]},
)

end_time = time.time()
execution_time = end_time - start_time
# NOTE: If first time creating endpoint, 404 error will be returned if endpoint is not full created yet.
print("Response status:", response.status_code)
print("Reponse text:", response.text)
print("Execution time:", execution_time, "seconds")

# COMMAND ----------
# another way to call the endpoint
""" If we have multiple records and multiple columns to query:
json={"dataframe_split": {"columns": ["id", "location"], "data": [["25", "New York"], ["26", "San Francisco"]]}},
"""

response = requests.post(
    f"{serving_endpoint}",
    headers={"Authorization": f"Bearer {token}"},
    json={"dataframe_split": {"columns": ["id"], "data": [["25"]]}},
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Test
# MAGIC Send multiple requests concurrently and calculate average reponse to measure latency

# COMMAND ----------
# Initialize variables
serving_endpoint = f"https://{host}/serving-endpoints/wine-quality-feature-serving/invocations"
id_list = preds_df.select("Id").rdd.flatMap(lambda x: x).collect()
headers = {"Authorization": f"Bearer {token}"}
num_requests = 10

# COMMAND ----------
# Function to make a request and record latency
def send_request():
    random_id = random.choice(id_list)
    start_time = time.time()
    response = requests.post(
        serving_endpoint,
        headers=headers,
        json={"dataframe_records": [{"id": random_id}]},
    )
    end_time = time.time()
    latency = end_time - start_time  # Calculate latency for this request
    return response.status_code, latency
# COMMAND ----------

# Measure total execution time
total_start_time = time.time()
latencies = []

# Send requests concurrently
with ThreadPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(send_request) for _ in range(num_requests)]

    for future in as_completed(futures):
        status_code, latency = future.result()
        latencies.append(latency)

total_end_time = time.time()
total_execution_time = total_end_time - total_start_time

# Calculate the average latency
average_latency = sum(latencies) / len(latencies)

print("\nTotal execution time:", total_execution_time, "seconds")
print("Average latency per request:", average_latency, "seconds") 