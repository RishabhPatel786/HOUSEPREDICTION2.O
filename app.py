from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model data with a safety fallback
model_data_path = 'model_data.pkl'
model, model_columns = None, []

if os.path.exists(model_data_path):
    with open(model_data_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        model_columns = data['columns']

def format_inr(amount):
    """Indian Currency Formatting (Lakhs/Crores)"""
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
    if not model:
        return render_template('index.html', prediction_text="Error: Model not found.")

    try:
        # Get and sanitize inputs
        area = float(request.form.get('area', 0))
        bhk = float(request.form.get('bhk', 0))
        age = float(request.form.get('age', 0))
        bath = float(request.form.get('bath', 0))
        location = request.form.get('location', 'City')

        # Logic: Area must be > 0
        if area <= 0:
            return render_template('index.html', prediction_text="Error: Area must be greater than 0.")

        # Build feature vector
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
        input_df['Area_SqFt'] = area
        input_df['BHK'] = bhk
        input_df['Age_Years'] = age
        input_df['Bathrooms'] = bath
        
        loc_col = f'Location_{location}'
        if loc_col in model_columns:
            input_df[loc_col] = 1

        # Predict
        prediction = model.predict(input_df)[0]
        
        # Calculate Per Sq. Ft. Rate
        per_sqft = prediction / area

        return render_template('index.html', 
                               prediction_text=f'₹{format_inr(prediction)}',
                               per_sqft=f'₹{format_inr(per_sqft)}/sq.ft',
                               location_name=location.replace('_', ' '),
                               area=area, bhk=int(bhk))

    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="Something went wrong. Please check inputs.")

if __name__ == "__main__":
    app.run(debug=True)