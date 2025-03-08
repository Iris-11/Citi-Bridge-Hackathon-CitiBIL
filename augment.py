import pandas as pd
import random
import numpy as np

df = pd.read_csv(r"C:\Users\vaish\Desktop\Citi-Hackathon\Backend\augmented_dataset.csv")

# Function to generate synthetic data
def generate_synthetic_data(existing_df, num_samples=100):
    synthetic_data = []
    for _ in range(num_samples):
        # Randomly sample a real record
        base_record = existing_df.sample(n=1).iloc[0]

        # Create variations for each column
        new_record = {
            "Business_ID": max(existing_df["Business_ID"]) + random.randint(1, 500),  # Ensure unique ID
            "Annual_Revenue (₹)": int(base_record["Annual_Revenue (₹)"] * random.uniform(0.8, 1.2)),  # ±20% variation
            "Loan_Amount (₹)": int(base_record["Loan_Amount (₹)"] * random.uniform(0.8, 1.2)),  # ±20% variation
            "GST_Compliance (%)": min(100, max(60, base_record["GST_Compliance (%)"] + random.randint(-10, 10))),  # 60-100%
            "Past_Defaults": random.choices([0, 1], weights=[0.8, 0.2])[0],  # 80% chance of no defaults
            "Bank_Transactions": random.choice(["Stable", "Unstable", "High Volume", "Low Volume"]),
            "Market_Trend": random.choice(["Growth", "Declining", "Stable"]),
        }

        # Credit Score Logic (Ensuring a Logical Score)
        base_score = base_record["Credit_Score"]
        if new_record["Past_Defaults"] == 1:
            new_record["Credit_Score"] = max(500, base_score - random.randint(50, 150))
        elif new_record["Market_Trend"] == "Declining":
            new_record["Credit_Score"] = max(550, base_score - random.randint(30, 100))
        else:
            new_record["Credit_Score"] = min(900, base_score + random.randint(-50, 50))

        synthetic_data.append(new_record)

    return pd.DataFrame(synthetic_data)

# Generate 100 synthetic records
synthetic_df = generate_synthetic_data(df, num_samples=100)

# Combine original and synthetic datasets
augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

# Save to a new CSV file
augmented_df.to_csv("augmented_dataset.csv", index=False)

print("✅ Augmented dataset saved as 'augmented_dataset.csv' with", len(augmented_df), "records.")
