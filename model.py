import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# 1. Load Data (Replace with your actual csv)
# Assume columns: 'Avg_Area_Income', 'House_Age', 'Rooms', 'Bedrooms', 'Population', 'Price'
df = pd.read_csv('housing_data.csv')

# 2. Features and Target
X = df[['Avg_Area_Income', 'House_Age', 'Rooms', 'Bedrooms', 'Population']]
y = df['Price']

# 3. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Save the model to a file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as model.pkl")