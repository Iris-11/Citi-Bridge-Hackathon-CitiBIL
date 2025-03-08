import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import r2_score, root_mean_squared_error, roc_auc_score, f1_score, mean_absolute_error
import numpy as np


# Load dataset
df = pd.read_csv(r'C:\Users\vaish\Desktop\Citi-Hackathon\Backend\augmented_dataset.csv')


# Define target variable
label = 'Credit_Score'


# Ensure label exists
if label not in df.columns:
    raise ValueError(f"Label '{label}' not found in dataset")


# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()


# Convert categorical variables to numeric using category codes
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes


# Fill missing values with median
df.fillna(df.median(), inplace=True)
df.drop(columns=['Business_ID'])
# Split data
X = df.drop(columns=[label])
y = df[label]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train AutoGluon
#autogluon_predictor = TabularPredictor(label=label, path="ag_model").fit(df)


# Train XGBoost
print("Training XGBoost model...")
#xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train)


# Evaluate XGBoost
xgb_preds = xgb_model.predict(X_test)
print(f"XGBoost MAE: {mean_absolute_error(y_test, xgb_preds):.2f}")
print(f"XGBoost RMSE: {np.sqrt(root_mean_squared_error(y_test, xgb_preds)):.2f}")
print(f"XGBoost R² Score: {r2_score(y_test, xgb_preds):.2f}")

# Save XGBoost model
with open("xgboost_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
print("✅ XGBoost model trained & saved!")


# Train Random Forest
print("Training Random Forest model...")
#rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
rf_model.fit(X_train, y_train)


# Evaluate Random Forest
rf_preds = rf_model.predict(X_test)
print(f"Random Forest MAE: {mean_absolute_error(y_test, rf_preds):.2f}")
print(f"Random Forest RMSE: {np.sqrt(root_mean_squared_error(y_test, rf_preds)):.2f}")
print(f"Random Forest R² Score: {r2_score(y_test, rf_preds):.2f}")


# Save Random Forest model
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("✅ Random Forest model trained & saved!")


# Convert data to float64 for SHAP
X_train_float = X_train.astype(np.float64)
X_test_float = X_test.astype(np.float64)


# Generate SHAP explanations
print("Generating SHAP explanations...")
explainer_xgb = shap.Explainer(xgb_model, X_train_float)
shap_values_xgb = explainer_xgb(X_test_float)


explainer_rf = shap.Explainer(rf_model, X_train_float)
shap_values_rf = explainer_rf(X_test_float)


# Save SHAP values
with open("shap_xgb.pkl", "wb") as f:
    pickle.dump(shap_values_xgb, f)


with open("shap_rf.pkl", "wb") as f:
    pickle.dump(shap_values_rf, f)


print("✅ SHAP explanations generated & saved!")