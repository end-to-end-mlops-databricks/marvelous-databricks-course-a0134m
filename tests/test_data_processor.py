import pandas as pd
import pytest

from src.wine_quality.data_processor import DataProcessor


@pytest.fixture
def sample_data():
    data = pd.DataFrame(
        {
            "fixed_acidity": [1, 1, 2, 2, 3, 4, 5],
            "volatile_acidity": [101, 102, 101, 103, 102, 5, 104],
            "citric_acid": [1, 2, 3, 4, 5.2, 6, 7],
            "residual_sugar": [1, 2, 3, 4, 5.2, 35, 40],
            "chlorides": [1, 2, 3, 4, 5.2, 6, 7],
            "free_sulfur_dioxide": [1, 2, 3, 4, 5.2, 6, 7],
            "total_sulfur_dioxide": [1, 2, 3, 4, 5.2, 6, 7],
            "density": [1, 2, 3, 4, 5.2, 6, 7],
            "pH": [1, 2, 3, 4, 5.2, 6, 7],
            "sulphates": [1, 2, 3, 4, 5.2, 6, 7],
            "alcohol": [1, 2, 3, 4, 5.2, 6, 7],
            "quality": [5, 4, 3, 2, 1, 3, 5],
            "id": [1, 2, 3, 4, 5, 6, 7],
        }
    )
    return data


@pytest.fixture
def data_processor(sample_data, project_config):
    return DataProcessor(sample_data, project_config)


@pytest.fixture(name="processed_df")
def test_preprocess(data_processor, project_config):
    data_processor.preprocess()
    return data_processor.df


def test_df_not_null(processed_df):
    assert not processed_df.isnull().values.any(), "DataFrame contains null values after preprocessing"


def test_columns_present(processed_df, project_config):
    assert all(
        col in processed_df.columns for col in project_config.num_features + [project_config.target, "id"]
    ), "Not all required columns are present after preprocessing"


def test_id_is_string(processed_df, column_name="id"):
    assert all(
        isinstance(x, str) for x in processed_df[column_name]
    ), f"Column '{column_name}' contains non-string values."


def test_split_data(data_processor):
    train_set, test_set = data_processor.split_data()
    assert len(train_set) > 0, "Training set is empty"
    assert len(test_set) > 0, "Test set is empty"


def test_project_config(project_config):
    assert project_config.num_features is not None, "num_features is not set in project_config"
    assert project_config.target is not None, "target is not set in project_config"
    assert project_config.catalog_name is not None, "catalog_name is not set in project_config"
    assert project_config.schema_name is not None, "schema_name is not set in project_config"
