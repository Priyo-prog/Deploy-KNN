import math

import numpy as np
from pandas import array
import pandas as pd
import pytest 

from classification_model.predict_class import make_prediction
from classification_model.processing.data_manager import load_dataset, data_preprocess
from classification_model.config.core import config



# Pytest can understand only the code under 'test' package which starts with 'test_' code
def test_make_prediction(sample_input_data):

    X_train, X_test, y_train, y_test = data_preprocess(dataframe=sample_input_data)

    result = make_prediction(input_data=X_test)
    predicted_result = pd.DataFrame(result)

    assert isinstance(predicted_result, pd.DataFrame)