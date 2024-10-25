import pandas as pd
from wine_quality.config import ProjectConfig
import datetime
from sklearn.model_selection import train_test_split
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from datetime import datetime
from pyspark.sql import SparkSession

class DataProcessor:
    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig):
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration

    def preprocess(self):
        """Preprocess the DataFrame stored in self.df"""
        # Handle numeric features
        num_features = self.config.num_features
        for col in num_features:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Extract target and relevant features
        target = self.config.target
        relevant_columns = num_features + [target] + ['id']
        self.df = self.df[relevant_columns]

    def split_data(self, test_size=0.2, random_state=42):
        """Split the DataFrame (self.df) into training and test sets."""
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set
    
    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame, spark: SparkSession):
        """Save the train and test sets into Databricks tables."""

        train_set_with_timestamp = spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))   
        
        test_set_with_timestamp = spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC"))

        train_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set")
        
        test_set_with_timestamp.write.mode("append").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set")

        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")
        
        spark.sql(f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
          "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);")    