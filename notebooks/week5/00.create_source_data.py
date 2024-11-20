import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from wine_quality.config import ProjectConfig

# Load configuration
config = ProjectConfig.from_yaml(config_path="../../project_config.yml")
catalog_name = config.catalog_name
schema_name = config.schema_name
spark = SparkSession.builder.getOrCreate()

# Load train and test sets
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()
combined_set = pd.concat([train_set, test_set], ignore_index=True)
existing_ids = set(int(id) for id in combined_set['id'])

# Define function to create synthetic data without random state
def create_synthetic_data(df, num_rows=100):
    synthetic_data = pd.DataFrame()
    
    for column in df.columns:
        # Treat float and int differently
            # if pd.api.types.is_numeric_dtype(df[column]) and column != 'id':
            #     mean, std = df[column].mean(), df[column].std()
            #     synthetic_data[column] = np.random.normal(mean, std, num_rows)
        if pd.api.types.is_float_dtype(df[column]):
            mean, std = df[column].mean(), df[column].std()
            synthetic_data[column] = np.random.normal(mean, std, num_rows)
        elif pd.api.types.is_integer_dtype(df[column]) and column != 'id':
            mean, std = df[column].mean(), df[column].std()
            synthetic_data[column] = np.random.normal(mean, std, num_rows).astype(int)
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            synthetic_data[column] = np.random.choice(df[column].unique(), num_rows, 
                                                      p=df[column].value_counts(normalize=True))
        
        elif pd.api.types.is_datetime64_any_dtype(df[column]):
            min_date, max_date = df[column].min(), df[column].max()
            if min_date < max_date:
                synthetic_data[column] = pd.to_datetime(
                    np.random.randint(min_date.value, max_date.value, num_rows)
                )
            else:
                synthetic_data[column] = [min_date] * num_rows
        
        else:
            synthetic_data[column] = np.random.choice(df[column], num_rows)
    
    # Making sure that generated IDs are unique and do not previously exist 
    new_ids = []
    i = max(existing_ids) + 1 if existing_ids else 1
    while len(new_ids) < num_rows:
        if i not in existing_ids:
            new_ids.append(i)  # Id needs to be string, but leaving it as int to match train/test set. Will convert to string later.
            #new_ids.append(str(i))  # Convert numeric ID to string
        i += 1
    synthetic_data['id'] = new_ids

    return synthetic_data

# Create synthetic data
synthetic_df = create_synthetic_data(combined_set)

# Create source_data table manually using Create table like train_set 
existing_schema = spark.table(f"{catalog_name}.{schema_name}.source_data").schema

synthetic_spark_df = spark.createDataFrame(synthetic_df, schema=existing_schema)

train_set_with_timestamp = synthetic_spark_df.withColumn(
    "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
)

# Append synthetic data as new data to source_data table
train_set_with_timestamp.write.mode("append").saveAsTable(
    f"{catalog_name}.{schema_name}.source_data"
)