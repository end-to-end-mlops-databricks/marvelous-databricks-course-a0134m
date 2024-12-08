"""
This script handles the deployment of a wine quality prediction model to a Databricks serving endpoint.
Key functionality:
- Loads project configuration from YAML
- Retrieves the model version from previous task values
- Updates the serving endpoint configuration with:
  - Model registry reference
  - Scale to zero capability
  - Workload sizing
  - Specific model version
The endpoint is configured for feature-engineered model serving with automatic scaling.
"""

import argparse

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from pyspark.dbutils import DBUtils
from pyspark.sql import SparkSession
from workspace_utils import check_if_endpoint_exists, get_env_config_file

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
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)

args = parser.parse_args()
root_path = args.root_path

config_file_name = get_env_config_file(args.env)
config_path = f"{root_path}/{config_file_name}"
# config_path = ("/Volumes/mlops_students/mahajan134/mlops_vol/project_config.yml")
config = ProjectConfig.from_yaml(config_path=config_path)
endpoint_name = "wine-quality-model-serving-fe"
model_name = "wine-quality-model-fe"

spark = SparkSession.builder.getOrCreate()
dbutils = DBUtils(spark)

model_version = dbutils.jobs.taskValues.get(taskKey="evaluate_model", key="model_version")

workspace = WorkspaceClient()

catalog_name = config.catalog_name
schema_name = config.schema_name

if check_if_endpoint_exists(workspace, endpoint_name):
    print(f"Endpoint '{endpoint_name}' exists.")
    workspace.serving_endpoints.update_config_and_wait(
        name=endpoint_name,
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog_name}.{schema_name}.{model_name}",
                scale_to_zero_enabled=True,
                workload_size="Small",
                entity_version=model_version,
            )
        ],
    )
else:
    print(f"Endpoint '{endpoint_name}' does not exist, creating new endpoint.")
    workspace.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=f"{catalog_name}.{schema_name}.{model_name}",
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                    entity_version=1,
                )
            ]
        ),
    )
