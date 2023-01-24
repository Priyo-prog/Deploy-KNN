import typing as t

import numpy as np
import pandas as pd

#from classification_model import __version__ as _version
from classification_model.config.core import config
from classification_model.processing.data_manager import load_model
#from classification_model.processing.validation import validate_inputs

model_file_name = f"{config.app_config.model_save_file}.pkl"
model = load_model(file_name=model_file_name)


def make_prediction(
    *,
    input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)
    #validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None}

   
    predictions = model.predict(
        #X=validated_data[config.model_config.features]
        X = data
    )
    results = {
        "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
        #"version": _version
    }

    return results