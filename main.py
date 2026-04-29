import os

print("Credit Card Fraud Detection System")
print("=" * 45)

print("\nStep 1: Generate dataset")
os.system("python src/generate_dataset.py")

print("\nStep 2: Train model")
os.system("python src/train_model.py")

print("\nStep 3: Create visualizations")
os.system("python src/visualize.py")

print("\nPipeline completed successfully!")