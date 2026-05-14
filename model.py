import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load Data
df = pd.read_csv('housing_data.csv')

# 2. Features (X) and Target (y)
# These must match the CSV column names exactly
X = df[['Area_SqFt', 'BHK', 'Age_Years', 'Bathrooms', 'Location_Tier', 'Quality_Score']]
y = df['Price']

# 3. Train the Model
model = LinearRegression()
model.fit(X, y)

# 4. Save the Model and the column names
model_data = {
    'model': model,
    'columns': X.columns.tolist()
}

with open('model_data.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model trained successfully with 6 features!")