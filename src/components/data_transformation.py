import sys 
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        """
        This function is responsible for data transformation.
        It creates preprocessing pipelines for numerical and categorical features.

        Returns
        -------
        sklearn.pipeline.Pipeline
            A pipeline object that applies the transformations.
        """
        try:
            logging.info("Data Transformation initiated")
            # Define which columns are numerical and which are categorical
            numerical_columns = ['writing score', 'reading score']
            categorical_columns = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
            # Numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            # Categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Numerical and categorical pipelines created successfully.")
            # Combine numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            ) 
            logging.info("Preprocessor object created successfully.")
            return preprocessor
        except Exception as e:
            logging.error("Error in data transformation: %s", e)
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        This function initiates the data transformation process.
        It reads the training and testing data, applies the transformations,
        and saves the preprocessor object.

        Parameters
        ----------
        train_path : str
            Path to the training data CSV file.
        test_path : str
            Path to the testing data CSV file.

        Returns
        -------
        tuple
            Transformed training and testing arrays, and the preprocessor object path.
        """
        try:
            # Read the training and testing data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Training and testing data read successfully.")

            logging.info("Obtaining preprocessor object.")
            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = 'math score'
            numerical_columns = ['writing score', 'reading score']

            # Separate input features and target variable for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target variable for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applied preprocessing object on training and testing data on dataframe.")
            
            # Apply transformations to training data
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]

            # Apply transformations to testing data
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Data transformation completed successfully.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )   
        except Exception as e:
            logging.error("Error in initiate_data_transformation: %s", e)
            raise CustomException(e, sys)   
            
if __name__ == "__main__":
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(
        train_path=r'artifacts\train.csv', 
        test_path=r'artifacts\test.csv'
    )
