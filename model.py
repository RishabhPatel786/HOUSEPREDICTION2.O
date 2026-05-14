import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

df = pd.read_csv('housing_data.csv')

# Convert text locations into numeric columns
df_encoded = pd.get_dummies(df, columns=['Location'])

# Define Features (X) and Target (y)
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

model = LinearRegression()
model.fit(X, y)

# Save both the model and the column names (needed for prediction)
model_data = {'model': model, 'columns': X.columns.tolist()}
with open('model_data.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Model trained with Location features!")