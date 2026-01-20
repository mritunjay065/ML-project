import os
import sys

from xgboost import train
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
	"""Simple data ingestion helper.

	Methods
	-------
	initiate_data_ingestion():
		Reads CSV from `file_path` and returns a pandas DataFrame.

	ingest_and_split(test_size=0.2, random_state=42, artifacts_dir='artifacts'):
		Reads the CSV, splits into train/test, saves to `artifacts_dir`, and
		returns the paths to the saved CSVs.
	"""
	train_data_path: str = os.path.join('artifacts', 'train.csv')
	test_data_path: str = os.path.join('artifacts', 'test.csv') 
	raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
	def __init__(self):
		self.ingestion_config = DataIngestionConfig()

	def initiate_data_ingestion(self, file_path: str) -> pd.DataFrame:
		"""Reads CSV from `file_path` and returns a pandas DataFrame.

		Parameters
		----------
		file_path : str
			Path to the CSV file.

		Returns
		-------
		pd.DataFrame
			DataFrame containing the data from the CSV.
		"""
		logging.info("Starting data ingestion from file: %s", file_path)
		try:
			# Use the provided file_path argument, default to Students.csv if not provided
			if not file_path:
				file_path = r'notebook\data\Students.csv'
			df = pd.read_csv(file_path)
			logging.info("Data ingestion completed successfully.")

			os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

			df.to_csv(self.ingestion_config.raw_data_path, index=False)

			logging.info("Raw data saved at: %s", self.ingestion_config.raw_data_path)


			logging.info("Train test split initiated.")	
			train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

			train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)

			test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

			logging.info("Ingestion of data is completed.")
			return(
				self.ingestion_config.train_data_path,
				self.ingestion_config.test_data_path
			)
		
		except Exception as e:
			logging.error("Error during data ingestion: %s", e)
			raise CustomException(e, sys)
		

if __name__ == "__main__":
	obj = DataIngestion()
	# Provide the default file path for Students.csv
	obj.initiate_data_ingestion(r'notebook\\data\\Students.csv')
	train_data, test_data = obj.initiate_data_ingestion(
		r'notebook\data\Students.csv'
	)
	data_transformation = DataTransformation()
	data_transformation.initiate_data_transformation(
		train_path=train_data,
		test_path=test_data
	)