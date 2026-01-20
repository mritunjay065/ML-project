"""
Training Pipeline for Student Performance Prediction Model
"""
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    """
    Complete training pipeline that orchestrates:
    1. Data Ingestion
    2. Data Transformation
    3. Model Training
    """
    
    def __init__(self):
        """Initialize the training pipeline."""
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def start_training(self, data_path: str = r'notebook\data\Students.csv'):
        """
        Execute the complete training pipeline.
        
        Parameters
        ----------
        data_path : str
            Path to the training data CSV file.
            
        Returns
        -------
        float
            R2 score of the best trained model.
        """
        try:
            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE STARTED")
            logging.info("=" * 60)
            
            # Step 1: Data Ingestion
            logging.info("\n[STEP 1/3] DATA INGESTION")
            logging.info("-" * 60)
            print("\n🔄 Step 1/3: Loading and splitting data...")
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion(data_path)
            logging.info(f"Training data: {train_data_path}")
            logging.info(f"Testing data: {test_data_path}")
            print("✅ Data ingestion completed!")
            
            # Step 2: Data Transformation
            logging.info("\n[STEP 2/3] DATA TRANSFORMATION")
            logging.info("-" * 60)
            print("\n🔄 Step 2/3: Preprocessing and transforming data...")
            train_arr, test_arr, preprocessor_path = self.data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )
            logging.info(f"Preprocessor saved: {preprocessor_path}")
            print("✅ Data transformation completed!")
            
            # Step 3: Model Training
            logging.info("\n[STEP 3/3] MODEL TRAINING")
            logging.info("-" * 60)
            print("\n🔄 Step 3/3: Training models with hyperparameter tuning...")
            print("   Training 7 regression models...")
            print("   This may take a few minutes...\n")
            
            r2_score = self.model_trainer.initiate_model_trainer(
                train_array=train_arr,
                test_array=test_arr
            )
            
            logging.info("=" * 60)
            logging.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logging.info(f"Best Model R2 Score: {r2_score:.4f}")
            logging.info("=" * 60)
            
            print("\n" + "=" * 60)
            print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"Best Model R2 Score: {r2_score:.4f} ({r2_score*100:.2f}% accuracy)")
            print("\nSaved Artifacts:")
            print(f"  📁 artifacts/model.pkl         - Trained model")
            print(f"  📁 artifacts/preprocessor.pkl  - Data preprocessor")
            print(f"  📁 artifacts/train.csv         - Training data")
            print(f"  📁 artifacts/test.csv          - Testing data")
            print("=" * 60 + "\n")
            
            return r2_score
            
        except Exception as e:
            logging.error("Training pipeline failed!")
            raise CustomException(e, sys)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" " * 15 + "TRAINING PIPELINE")
    print("=" * 60)
    
    # Create and run pipeline
    pipeline = TrainingPipeline()
    r2_score = pipeline.start_training()
    
    print("\n💡 Next Steps:")
    print("   • Run 'python test_pipeline.py' to test predictions")
    print("   • Run 'python show_results.py' to see detailed results")
    print("=" * 60 + "\n")
