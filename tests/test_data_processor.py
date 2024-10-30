import pandas as pd
import pytest


@pytest.fixture
def sample_data():
    return pd.DataFrame({"num1": [1, 2, 3, 4, 5], "num2": [5, 4, 3, 2, 1], "target": [10, 20, 30, 40, 50]})
