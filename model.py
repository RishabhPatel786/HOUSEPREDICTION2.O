import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load Data
df = pd.read_csv('housing_data.csv')

# 2. Encode Locations (Creates columns like Location_City, Location_Village)
df_encoded = pd.get_dummies(df, columns=['Location'])

# 3. Separate Features and Target
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# 4. Train
model = LinearRegression()
model.fit(X, y)

# 5. Save Model AND Column Names (Crucial for app.py to match the order)
model_data = {
    'model': model,
    'columns': X.columns.tolist()
}

with open('model_data.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model retrained with location awareness!")