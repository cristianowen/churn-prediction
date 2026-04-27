import pandas as pd

# --------Load data---------
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Drop useless column
df = df.drop("customerID", axis=1)

# Convert target variable to binary (Churn: No=0, Yes=1) so ML can understand it
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Convert TotalCharges to numeric, coercing errors to NaN (e.g., for empty strings)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Remove missing values
df = df.dropna()

# Convert categorical variables into numeric using one-hot encoding (get_dummies)
df = pd.get_dummies(df, drop_first=True)

# Check shape
print("Shape:", df.shape)

# ========================
# MACHINE LEARNING MODEL
# ========================

print("Starting model...")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

import numpy as np

# Get feature importance (coefficients)
coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
})

# Sort by importance
coefficients = coefficients.sort_values(by="Importance", ascending=False)

print(coefficients.head(10))