import numpy as np
import pandas as pd
from config.core import config
from classification_model.predict_class import make_prediction
from processing.data_manager import load_dataset, data_preprocess
from config.core import PREDICTED_RESULT_DIR


def run_prediction():
    data = load_dataset(file_name=config.app_config.training_data_file)

    X_train, X_test, y_train, y_test = data_preprocess(dataframe=data)

    df = pd.DataFrame(make_prediction(input_data=X_test))

    save_file_name = f"{config.app_config.predict_data_file}.csv"

    df.to_csv(PREDICTED_RESULT_DIR/save_file_name)

if __name__ == "__main__":
    run_prediction()



