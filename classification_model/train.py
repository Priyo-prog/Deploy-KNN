import numpy as np
from config.core import config
from processing.data_manager import load_dataset, save_model, data_preprocess
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def run_training() -> None:
    """ Train the model """

    # Read the training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    X_train, X_test, y_train, y_test = data_preprocess(dataframe=data)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     data[config.model_config.features],  # predictors
    #     data[config.model_config.target],
    #     test_size=config.model_config.test_size,
    #     # we are setting the random seed here
    #     # for reproducibility
    #     random_state=config.model_config.random_state,
    # )
    # y_train = np.log(y_train)

    # Fit the model
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)

    # persist trained model
    save_model(model_fit=classifier)


if __name__ == "__main__":
    run_training()

