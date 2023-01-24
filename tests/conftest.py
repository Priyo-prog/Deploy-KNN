import pytest

from classification_model.config.core import config
from classification_model.processing.data_manager import load_dataset

@pytest.fixture()
def sample_input_data():
    return load_dataset(filename=config.app_config.training_data_file)