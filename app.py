from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load model data
model_path = 'model_data.pkl'
model, model_columns = None, []
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model, model_columns = data['model'], data['columns']

def format_inr(amount):
    amount = max(100000, round(amount))
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
    try:
        area = float(request.form.get('area'))
        bhk = float(request.form.get('bhk'))
        location = request.form.get('location')
        
        # Prepare Data
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
        input_df['Area_SqFt'], input_df['BHK'] = area, bhk
        input_df['Age_Years'] = float(request.form.get('age'))
        input_df['Bathrooms'] = float(request.form.get('bath'))
        
        loc_col = 'Location_' + location
        if loc_col in model_columns: input_df[loc_col] = 1

        # Prediction
        raw_val = model.predict(input_df)[0]
        formatted_price = format_inr(raw_val)
        per_sqft = format_inr(raw_val / area)

        return render_template('index.html', 
                               prediction_text=f'₹{formatted_price}',
                               per_sqft=f'₹{per_sqft}/sq.ft',
                               location=location.replace('_', ' '))
    except:
        return render_template('index.html', prediction_text="Check Input Data")

if __name__ == "__main__":
    app.run(debug=True)