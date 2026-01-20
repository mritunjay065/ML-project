import pickle
import pandas as pd
import numpy as np

# Load the trained model and preprocessor
with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

print("=" * 60)
print("STUDENT PERFORMANCE PREDICTION SYSTEM")
print("=" * 60)

# Example: Create a sample student data for prediction
sample_data = pd.DataFrame({
    'gender': ['male'],
    'race/ethnicity': ['group B'],
    'parental level of education': ["bachelor's degree"],
    'lunch': ['standard'],
    'test preparation course': ['none'],
    'reading score': [72],
    'writing score': [74]
})

print("\nSample Student Data:")
print(sample_data.to_string(index=False))

# Transform the data using the preprocessor
data_scaled = preprocessor.transform(sample_data)

# Make prediction
prediction = model.predict(data_scaled)

print("\n" + "=" * 60)
print(f"PREDICTED MATH SCORE: {prediction[0]:.2f}")
print("=" * 60)

# You can also test with the test dataset
print("\n\nTesting on Test Dataset...")
test_df = pd.read_csv('artifacts/test.csv')

# Separate features and target
X_test = test_df.drop(['math score'], axis=1)
y_test = test_df['math score']

# Transform and predict
X_test_scaled = preprocessor.transform(X_test)
predictions = model.predict(X_test_scaled)

# Show first 10 predictions vs actual
print("\nFirst 10 Predictions vs Actual:")
print("-" * 60)
results_df = pd.DataFrame({
    'Actual Math Score': y_test.head(10).values,
    'Predicted Math Score': predictions[:10],
    'Difference': y_test.head(10).values - predictions[:10]
})
print(results_df.to_string(index=False))

# Calculate accuracy metrics
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print("\n" + "=" * 60)
print("MODEL PERFORMANCE ON TEST DATA:")
print("=" * 60)
print(f"R2 Score: {r2:.4f} ({r2*100:.2f}% accuracy)")
print(f"Average Error (MAE): {mae:.2f} points")
print(f"RMSE: {rmse:.2f} points")
print("=" * 60)
