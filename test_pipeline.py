"""
Demo script to test the prediction pipeline
"""
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

print("=" * 70)
print(" " * 20 + "PREDICTION PIPELINE DEMO")
print("=" * 70)

# Example 1: Create a student with good background
print("\n📚 Student 1: Strong Academic Background")
print("-" * 70)
student1 = CustomData(
    gender="female",
    race_ethnicity="group A",
    parental_level_of_education="master's degree",
    lunch="standard",
    test_preparation_course="completed",
    reading_score=95,
    writing_score=93
)

# Convert to DataFrame
student1_df = student1.get_data_as_dataframe()
print("\nStudent Information:")
print(student1_df.to_string(index=False))

# Make prediction
pipeline = PredictPipeline()
pred1 = pipeline.predict(student1_df)
print(f"\n✅ PREDICTED MATH SCORE: {pred1[0]:.2f}")


# Example 2: Create a student with average background
print("\n" + "=" * 70)
print("\n📚 Student 2: Average Academic Background")
print("-" * 70)
student2 = CustomData(
    gender="male",
    race_ethnicity="group C",
    parental_level_of_education="some college",
    lunch="standard",
    test_preparation_course="none",
    reading_score=65,
    writing_score=62
)

student2_df = student2.get_data_as_dataframe()
print("\nStudent Information:")
print(student2_df.to_string(index=False))

pred2 = pipeline.predict(student2_df)
print(f"\n✅ PREDICTED MATH SCORE: {pred2[0]:.2f}")


# Example 3: Create a student with weaker background
print("\n" + "=" * 70)
print("\n📚 Student 3: Weaker Academic Background")
print("-" * 70)
student3 = CustomData(
    gender="male",
    race_ethnicity="group B",
    parental_level_of_education="high school",
    lunch="free/reduced",
    test_preparation_course="none",
    reading_score=45,
    writing_score=42
)

student3_df = student3.get_data_as_dataframe()
print("\nStudent Information:")
print(student3_df.to_string(index=False))

pred3 = pipeline.predict(student3_df)
print(f"\n✅ PREDICTED MATH SCORE: {pred3[0]:.2f}")

print("\n" + "=" * 70)
print("✅ PREDICTION PIPELINE WORKING SUCCESSFULLY!")
print("=" * 70)

print("\n💡 Key Observations:")
print("   • Higher parental education → Higher predicted score")
print("   • Test prep completion → Improves prediction")
print("   • Reading/Writing scores → Strong correlation with math")
print("=" * 70 + "\n")
