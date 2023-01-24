import typing as t
from pathlib import Path

import joblib
import pandas as pd

#from classification_model import __version__ as _version

from classification_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def data_preprocess(*, dataframe:pd.DataFrame):
     
    # Create features and labels 
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values

    # Create the Train Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=config.model_config.test_size,
        random_state=config.model_config.random_state
    )

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


def save_model(*, model_fit) -> None:
    #save_file_name = f"{config.app_config.model_save_file}{_version}.pkl"
    save_file_name = f"{config.app_config.model_save_file}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    #remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(model_fit, save_path)


def load_model(*, file_name: str) -> None:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model    