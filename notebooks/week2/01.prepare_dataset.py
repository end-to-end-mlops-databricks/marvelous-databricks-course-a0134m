# Databricks notebook source
from wine_quality.data_processor import DataProcessor
from wine_quality.config import ProjectConfig
from datetime import datetime
import pandas as pd
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

config = ProjectConfig.from_yaml(config_path="../../project_config.yml")

# COMMAND ----------
# Load the dataset
df = spark.read.csv(
    "/Volumes/mlops_students/mahajan134/mlops_vol/wine_quality.csv",
    header=True,
    inferSchema=True).toPandas()

# COMMAND ----------
data_processor = DataProcessor(pandas_df=df, config=config)
data_processor.preprocess()
train_set, test_set = data_processor.split_data()
data_processor.save_to_catalog(train_set=train_set, test_set=test_set, spark=spark)