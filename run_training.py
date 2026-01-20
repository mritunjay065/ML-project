"""
Quick script to run the complete training pipeline
"""
from src.pipeline.train_pipeline import TrainingPipeline

if __name__ == "__main__":
    # Create and run the training pipeline
    pipeline = TrainingPipeline()
    pipeline.start_training()
