import logging

import pandas as pd

from wine_quality.config import ProjectConfig
from wine_quality.data_processor import DataProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load configuration
# with open("project_config.yml", "r") as file:
#     config = yaml.safe_load(file)

# print("Configuration loaded:")
# print(yaml.dump(config, default_flow_style=False))
# data_processor = DataProcessor("data/wine_quality.csv", config)

# Initialize DataProcessor
config = ProjectConfig.from_yaml(config_path="project_config.yml")

df = pd.read_csv("data/wine_quality.csv")

data_processor = DataProcessor(pandas_df=df, config=config)
logger.info("DataProcessor initialized.")

# Preprocess the data
data_processor.preprocess()
logger.info("Data preprocessed.")
