# 🎓 Student Performance Predictor

A complete end-to-end machine learning project that predicts student math scores based on various demographic and academic factors. Built with Flask web interface and professional ML pipelines.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **88.04% Prediction Accuracy** using Linear Regression
- **7 ML Algorithms** with automated hyperparameter tuning
- **Professional Pipelines** for training and prediction
- **Modern Web Interface** built with Flask
- **Comprehensive Metrics** (R2, MAE, RMSE, MSE)
- **Production-Ready** code structure

## 📊 Model Performance

| Model | Test R2 Score | Hyperparameters |
|-------|--------------|-----------------|
| **Linear Regression** 🏆 | **0.8804** | Default |
| Gradient Boosting | 0.8748 | lr=0.05, n_estimators=128 |
| AdaBoost | 0.8523 | lr=0.5, n_estimators=256 |
| Random Forest | 0.8534 | n_estimators=256 |
| XGBoost | 0.8492 | lr=0.05, n_estimators=64 |
| Decision Tree | 0.7293 | criterion='friedman_mse' |
| K-Neighbors | 0.5197 | n_neighbors=11 |

**Best Model Metrics:**
- R2 Score: 0.8804 (88.04% accuracy)
- MAE: 4.21 points
- RMSE: 5.39 points

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Conda (recommended) or pip

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd "ML project"

# Create virtual environment
conda create -p env python=3.8 -y
conda activate env/

# Install dependencies
pip install -r requirements.txt
```

### Run the Web Application

```bash
# Start Flask server
python app.py

# Open browser to http://127.0.0.1:5000
```

### Train the Model

```bash
# Run complete training pipeline
python run_training.py
```

### Make Predictions

```bash
# Test prediction pipeline
python test_pipeline.py

# View model results
python show_results.py
```

## 📁 Project Structure

```
ML project/
├── app.py                      # Flask web application
├── src/
│   ├── pipeline/
│   │   ├── train_pipeline.py   # Training pipeline
│   │   └── predict_pipeline.py # Prediction pipeline
│   ├── components/
│   │   ├── data_ingestion.py   # Data loading & splitting
│   │   ├── data_transformation.py  # Feature preprocessing
│   │   └── model_trainer.py    # Model training & tuning
│   └── utils.py                # Utility functions
├── templates/
│   ├── index.html              # Landing page
│   └── home.html               # Prediction form
├── static/
│   └── favicon.png             # App icon
├── artifacts/                  # Saved models & data
├── notebook/                   # Jupyter notebooks
└── requirements.txt            # Dependencies
```

## 🔄 ML Pipeline

### Training Pipeline
1. **Data Ingestion** - Load and split dataset (80/20)
2. **Data Transformation** - Preprocess features (scaling, encoding)
3. **Model Training** - Train 7 models with GridSearchCV
4. **Model Selection** - Save best performing model

### Prediction Pipeline
1. **Load Model** - Load trained model and preprocessor
2. **Transform Input** - Preprocess new data
3. **Predict** - Generate math score prediction

## 🌐 Web Application

The Flask web app provides an intuitive interface for predictions:

- **Landing Page** - Modern gradient design with feature highlights
- **Prediction Form** - Input student information
- **Real-time Results** - Instant math score predictions

**Input Features:**
- Gender
- Race/Ethnicity
- Parental Level of Education
- Lunch Type
- Test Preparation Course
- Reading Score
- Writing Score

## 🛠️ Tech Stack

- **Backend**: Python, Flask
- **ML Libraries**: scikit-learn, XGBoost, pandas, numpy
- **Frontend**: HTML, CSS (custom styling)
- **Model Persistence**: dill/pickle
- **Version Control**: Git

## 📈 Dataset

The model is trained on student performance data with 1000 samples containing:
- Demographic information
- Academic background
- Test scores (reading, writing, math)

## 🎯 Use Cases

- **Educational Institutions** - Identify students who may need additional support
- **Parents & Teachers** - Predict math performance based on other scores
- **Researchers** - Analyze factors affecting student performance

## 🚀 Deployment

Ready to deploy on:
- Render
- Railway
- Google Cloud Run
- PythonAnywhere
- Heroku

## 📝 API Usage

```python
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

# Create student data
student = CustomData(
    gender="female",
    race_ethnicity="group A",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=85,
    writing_score=88
)

# Get prediction
pipeline = PredictPipeline()
prediction = pipeline.predict(student.get_data_as_dataframe())
print(f"Predicted Math Score: {prediction[0]:.2f}")
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Author

Created as a complete end-to-end ML project demonstrating:
- Data preprocessing
- Model training & selection
- Hyperparameter tuning
- Web application development
- Production-ready code structure

---

⭐ **Star this repo if you find it useful!**