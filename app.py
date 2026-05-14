from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the model data (which includes the model and the expected column order)
model_path = 'model_data.pkl'
model = None
model_columns = []

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        model_columns = data['columns']

def format_indian_currency(amount):
    """Formats number to Indian numbering system and prevents negative values"""
    amount = max(0, round(amount)) 
    s = str(amount)
    if len(s) <= 3: return s
    last_three = s[-3:]
    remaining = s[:-3]
    remaining = ",".join([remaining[max(i-2, 0):i] for i in range(len(remaining), 0, -2)][::-1])
    return f"{remaining},{last_three}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: model_data.pkl not found.")
    
    try:
        # 1. Get basic features
        area = float(request.form.get('area'))
        bhk = float(request.form.get('bhk'))
        age = float(request.form.get('age'))
        bath = float(request.form.get('bath'))
        location = request.form.get('location')

        # 2. Create a DataFrame with all zeros matching the training columns
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
        
        # 3. Fill numerical values
        input_df['Area_SqFt'] = area
        input_df['BHK'] = bhk
        input_df['Age_Years'] = age
        input_df['Bathrooms'] = bath
        
        # 4. Handle One-Hot Encoding for Location
        loc_col = 'Location_' + location
        if loc_col in model_columns:
            input_df[loc_col] = 1

        # 5. Predict
        prediction = model.predict(input_df)
        formatted_price = format_indian_currency(prediction[0])

        return render_template('index.html', 
                               prediction_text=f'₹{formatted_price}',
                               area=area, bhk=int(bhk))
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Please check inputs.")

if __name__ == "__main__":
    app.run(debug=True)