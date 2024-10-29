# Databricks notebook source
import json

import mlflow

mlflow.set_tracking_uri("databricks")

experiment_name = "/Shared/wine-quality-basic"
repo_name = "wine_quality"
git_sha_id = "f6ee5171f4bc705b755af1ce4665cfdf98901e73"

mlflow.set_experiment(experiment_name=experiment_name)
mlflow.set_experiment_tags({"repository_name": repo_name})



# COMMAND ----------
experiments = mlflow.search_experiments(
    filter_string=f"tags.repository_name='{repo_name}'"
)
print(experiments)

# COMMAND ----------
with open("mlflow_experiment.json", "w") as json_file:
    json.dump(experiments[0].__dict__, json_file, indent=4)

# COMMAND ----------
with mlflow.start_run(
    run_name="demo-run",
    tags={"git_sha": git_sha_id,
          "branch": "week2"},
    description="demo run",
) as run:
    mlflow.log_params({"type": "demo"})
    mlflow.log_metrics({"metric1": 1.0, "metric2": 2.0})
# COMMAND ----------
run_id = mlflow.search_runs(
    experiment_names=[experiment_name],
    filter_string=f"tags.git_sha='{git_sha_id}'",
).run_id[0]
run_info = mlflow.get_run(run_id=f"{run_id}").to_dictionary()
print(run_info)

# COMMAND ----------
with open("run_info.json", "w") as json_file:
    json.dump(run_info, json_file, indent=4)

# COMMAND ----------
print(run_info["data"]["metrics"])

# COMMAND ----------
print(run_info["data"]["params"])