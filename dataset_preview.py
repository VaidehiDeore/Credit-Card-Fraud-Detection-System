import pandas as pd
import os

# Load dataset
df = pd.read_csv("data/transactions.csv")

# Show first 10 rows
print(df.head(10))

# Save as image-like CSV preview
os.makedirs("images", exist_ok=True)
df.head(10).to_csv("images/dataset_preview.csv", index=False)

print("Preview saved at images/dataset_preview.csv")