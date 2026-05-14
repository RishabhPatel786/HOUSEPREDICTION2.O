import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load the updated property-only data
df = pd.read_csv('housing_data.csv')

# Features: Square Feet, BHK, Age, Bathrooms
X = df[['Area_SqFt', 'BHK', 'Age_Years', 'Bathrooms']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Property-focused model trained successfully!")