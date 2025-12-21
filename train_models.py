import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load data (exactly as notebook)
df = pd.read_csv('user_data_with_modified_temperature.csv')
df.drop(columns=['User_ID'], axis=1, inplace=True)
df['Gender'] = df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

# Features and target (exactly as notebook)
X = df.drop(columns=['Calories'], axis=1)
y = df['Calories']

# THREE-way split (exactly as notebook cell #16)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create models directory
os.makedirs('models', exist_ok=True)

# Train models (exactly as notebook cell #39)
print("Training Linear Regression...")
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
joblib.dump(linear_model, 'models/linear_regression.pkl')

print("Training Random Forest...")
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, 'models/random_forest.pkl')

print("Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
joblib.dump(gb_model, 'models/gradient_boosting.pkl')

print("Training XGBoost...")
xgb_model = XGBRegressor(random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, 'models/xgboost.pkl')

print("Training LightGBM...")
lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_model.fit(X_train, y_train)
joblib.dump(lgb_model, 'models/lightgbm.pkl')

# Save scaler
joblib.dump(scaler, 'models/scaler.pkl')
print("✓ All models trained and saved!")
