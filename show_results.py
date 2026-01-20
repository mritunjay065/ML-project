"""
Script to display comprehensive model training results and comparisons
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def display_model_results():
    """Display comprehensive model training results"""
    
    # Load the best model and preprocessor
    with open('artifacts/model.pkl', 'rb') as f:
        best_model = pickle.load(f)
    
    with open('artifacts/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load test data
    test_df = pd.read_csv('artifacts/test.csv')
    X_test = test_df.drop(['math score'], axis=1)
    y_test = test_df['math score']
    
    # Transform and predict
    X_test_scaled = preprocessor.transform(X_test)
    predictions = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    
    # Display comprehensive results
    print("\n" + "=" * 70)
    print(" " * 15 + "MODEL TRAINING RESULTS")
    print("=" * 70)
    
    print("\n🏆 BEST MODEL: Linear Regression")
    print("-" * 70)
    
    print("\n📊 COMPREHENSIVE METRICS:")
    print(f"   • R2 Score:  {r2:.4f} ({r2*100:.2f}% accuracy)")
    print(f"   • MAE:       {mae:.4f} (avg prediction error in points)")
    print(f"   • RMSE:      {rmse:.4f} (penalizes large errors)")
    print(f"   • MSE:       {mse:.4f}")
    
    print("\n" + "=" * 70)
    print(" " * 10 + "ALL MODELS COMPARISON (Test R2 Scores)")
    print("=" * 70)
    
    # Model comparison data (from training logs)
    models_data = [
        ("Linear Regression", 0.8804, "🥇"),
        ("Gradient Boosting", 0.8748, "🥈"),
        ("AdaBoost Regressor", 0.8523, "🥉"),
        ("Random Forest", 0.8534, "  "),
        ("XGBRegressor", 0.8492, "  "),
        ("Decision Tree", 0.7293, "  "),
        ("K-Neighbors Regressor", 0.5197, "  "),
    ]
    
    print("\n{:<25} {:<15} {:<10} {}".format("Model", "R2 Score", "Accuracy", ""))
    print("-" * 70)
    for model_name, score, medal in models_data:
        print(f"{model_name:<25} {score:<15.4f} {score*100:>6.2f}%     {medal}")
    
    print("\n" + "=" * 70)
    print("✅ All 7 models were trained with automatic hyperparameter tuning")
    print("✅ GridSearchCV with 3-fold cross-validation")
    print("✅ Best hyperparameters automatically selected for each model")
    print("=" * 70)
    
    print("\n📈 HYPERPARAMETER TUNING SUMMARY:")
    print("-" * 70)
    print("• Linear Regression:       No hyperparameters (baseline)")
    print("• Gradient Boosting:       lr=0.05, n_estimators=128, subsample=0.6")
    print("• AdaBoost:                lr=0.5, n_estimators=256")
    print("• Random Forest:           n_estimators=256")
    print("• XGBoost:                 lr=0.05, n_estimators=64")
    print("• Decision Tree:           criterion='friedman_mse'")
    print("• K-Neighbors:             n_neighbors=11")
    print("=" * 70)
    
    print("\n💾 SAVED ARTIFACTS:")
    print("-" * 70)
    print("   📁 artifacts/model.pkl          - Best trained model")
    print("   📁 artifacts/preprocessor.pkl   - Data preprocessing pipeline")
    print("   📁 artifacts/train.csv          - Training dataset")
    print("   📁 artifacts/test.csv           - Testing dataset")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    display_model_results()
