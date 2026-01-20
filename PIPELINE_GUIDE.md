# ML Project - Complete Guide

## 🚀 Quick Start

### **Train the Model**
```powershell
python run_training.py
```

### **Make Predictions**
```powershell
python test_pipeline.py
```

### **View Results**
```powershell
python show_results.py
```

---

## 📦 Project Structure

```
ML project/
├── src/
│   ├── pipeline/
│   │   ├── train_pipeline.py    # Training Pipeline
│   │   └── predict_pipeline.py  # Prediction Pipeline
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   └── utils.py
├── artifacts/                    # Saved models & data
├── run_training.py              # Run training pipeline
├── test_pipeline.py             # Test predictions
└── show_results.py              # Show model results
```

---

## 🔄 Pipelines Explained

### **Training Pipeline** (`TrainingPipeline`)
**Purpose**: Train the model from scratch

**What it does:**
1. Loads raw data
2. Splits into train/test
3. Applies preprocessing
4. Trains 7 models with hyperparameter tuning
5. Saves best model

**Usage:**
```python
from src.pipeline.train_pipeline import TrainingPipeline

pipeline = TrainingPipeline()
r2_score = pipeline.start_training()
```

---

### **Prediction Pipeline** (`PredictPipeline`)
**Purpose**: Make predictions with trained model

**What it does:**
1. Loads saved model & preprocessor
2. Transforms new data
3. Returns predictions

**Usage:**
```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

student = CustomData(
    gender="female",
    race_ethnicity="group A",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=85,
    writing_score=88
)

pipeline = PredictPipeline()
prediction = pipeline.predict(student.get_data_as_dataframe())
```

---

## 📊 Model Performance

- **Best Model**: Linear Regression
- **R2 Score**: 0.8804 (88.04% accuracy)
- **MAE**: 4.21 points
- **RMSE**: 5.39 points

---

## 🎯 When to Use Each Pipeline

| Pipeline | Use When |
|----------|----------|
| **Training** | Need to retrain model with new data |
| **Prediction** | Need to predict for new students |
