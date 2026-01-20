import pandas as pd
import sys
import dill

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

