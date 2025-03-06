# data_processing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load Dataset
df = pd.read_csv('FAOSTAT_data.csv')

# Data Cleaning & Preprocessing
df.dropna(inplace=True)
df = df[['Area', 'Item', 'Year', 'Element', 'Value']]

# Pivot table to reshape data
df = df.pivot_table(index=['Area', 'Item', 'Year'], columns='Element', values='Value').reset_index()

# Rename columns dynamically
column_mapping = {
    'Area harvested': 'Area_Harvested',
    'Production': 'Production'
}
df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)

# **Filter dataset for crop-related records**
df = df[['Area', 'Item', 'Year', 'Area_Harvested', 'Production']].dropna()

# Ensure 'Yield' is computed
if 'Production' in df.columns and 'Area_Harvested' in df.columns:
    df['Yield'] = df['Production'] / df['Area_Harvested']

# Final check for required columns
required_cols = ['Area_Harvested', 'Yield', 'Production']
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing expected columns: {missing_cols}")

# **Check dataset size before splitting**
if df.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing. Ensure filtering criteria are correct.")

# Splitting Data for Model Training
X = df[['Area_Harvested', 'Yield', 'Year']]
y = df['Production']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model and Scaler
joblib.dump(model, 'crop_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Model Evaluation
y_pred = model.predict(X_test)
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R2 Score: {r2_score(y_test, y_pred)}')
