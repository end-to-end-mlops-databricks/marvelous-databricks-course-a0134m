import sys
from pathlib import Path

import pytest
from pyspark.sql import SparkSession

from src.wine_quality.config import ProjectConfig


@pytest.fixture(scope="session")
def spark_session() -> SparkSession:
    if "pytest" not in sys.argv[0]:
        return SparkSession.builder.getOrCreate()


@pytest.fixture
def project_config():
    config_path = Path(__file__).parent / "test_config.yml"
    return ProjectConfig.from_yaml(config_path)
