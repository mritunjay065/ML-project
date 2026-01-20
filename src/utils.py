import pandas as pd
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging 


def save_object(file_path: str, obj: object) -> None:
    """Saves a Python object to a file using pickle.

    Parameters
    ----------
    file_path : str
        The path where the object should be saved.
    obj : object
        The Python object to be saved.

    Returns
    -------
    None
    """
    try:
        import pickle
        import os

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            logging.info("Object saved successfully at: %s", file_path)
            logging.info("Applied preprocessing object on training and testing data on dataframe.")
    except Exception as e:
        logging.error("Failed to save object at %s: %s", file_path, str(e))
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Loads a Python object from a file using pickle.

    Parameters
    ----------
    file_path : str
        The path to the saved object file.

    Returns
    -------
    object
        The loaded Python object.
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logging.error("Failed to load object from %s: %s", file_path, str(e))
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate multiple models with hyperparameter tuning using GridSearchCV.

    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training target.
    X_test : array-like
        Testing features.
    y_test : array-like
        Testing target.
    models : dict
        Dictionary of model names and model objects.
    param : dict
        Dictionary of hyperparameters for each model.

    Returns
    -------
    dict
        Dictionary containing model names as keys and test R2 scores as values.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = param[model_name]

            logging.info(f"\n{'=' * 50}")
            logging.info(f"Training {model_name}...")
            
            if para:  # Only use GridSearchCV if there are params to tune
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                
                best_params = gs.best_params_
                logging.info(f"Best Hyperparameters: {best_params}")
                
                model.set_params(**best_params)
                model.fit(X_train, y_train)
            else:
                # No hyperparameters to tune
                model.fit(X_train, y_train)
                logging.info(f"No hyperparameters to tune for {model_name}")

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

            logging.info(f"Train R2: {train_model_score:.4f} | Test R2: {test_model_score:.4f}")
            logging.info(f"{'=' * 50}")

        return report

    except Exception as e:
        raise CustomException(e, sys)
