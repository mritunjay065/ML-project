"""
Prediction Pipeline for Student Performance Prediction
"""
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    """
    Pipeline for making predictions using trained model and preprocessor.
    """
    
    def __init__(self):
        """Initialize the prediction pipeline."""
        pass

    def predict(self, features):
        """
        Make predictions on input features.
        
        Parameters
        ----------
        features : pd.DataFrame
            Input features for prediction.
            
        Returns
        -------
        array
            Predicted values.
        """
        try:
            # Load model and preprocessor
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            print("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Model and preprocessor loaded successfully!")
            
            # Transform features
            print("Transforming input data...")
            data_scaled = preprocessor.transform(features)
            
            # Make predictions
            print("Making predictions...")
            preds = model.predict(data_scaled)
            
            return preds
            
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Class for handling custom input data for predictions.
    Maps user input to the format expected by the model.
    """
    
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int):
        """
        Initialize custom data object.
        
        Parameters
        ----------
        gender : str
            Student's gender (male/female)
        race_ethnicity : str
            Student's race/ethnicity group (group A/B/C/D/E)
        parental_level_of_education : str
            Parent's education level
        lunch : str
            Lunch type (standard/free or reduced)
        test_preparation_course : str
            Test prep course completion (none/completed)
        reading_score : int
            Reading test score (0-100)
        writing_score : int
            Writing test score (0-100)
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        """
        Convert custom data to pandas DataFrame.
        
        Returns
        -------
        pd.DataFrame
            Data in DataFrame format ready for prediction.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
