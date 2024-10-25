from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("mlops-dev-cli").getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)
