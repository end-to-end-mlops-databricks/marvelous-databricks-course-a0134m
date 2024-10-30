import pytest
import pandas as pd
import numpy as np
from wine_quality.config import ProjectConfig
from wine_quality.data_processor  import DataProcessor


@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'num1': [1, 2, 3, 4, 5],
        'num2': [5, 4, 3, 2, 1],
        'target': [10, 20, 30, 40, 50]
    })