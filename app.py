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
        model = data['model']
        model_columns = data['columns']

def format_inr(amount):
    """Formats number to Indian numbering system (Lakhs/Crores)"""
    amount = max(0, round(amount))
    s = str(amount)
    if len(s) <= 3: return s
    last_three = s[-3:]
    remaining = s[:-3]
    remaining = ",".join([remaining[max(i-2, 0):i] for i in range(len(remaining), 0, -2)][::-1])
    return f"{remaining},{last_three}"

@app.route('/')
def home():
    return render_template('index.html', old_inputs={})

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction_text="Error: Model missing.", old_inputs={})

    try:
        # 1. Capture Inputs
        inputs = {
            'area': request.form.get('area'),
            'bhk': request.form.get('bhk'),
            'age': request.form.get('age'),
            'bath': request.form.get('bath'),
            'quality': request.form.get('quality'),
            'location': request.form.get('location')
        }

        # 2. Logic & Feature Engineering
        tier_map = {'City': 4, 'Main_Road': 3, 'Outskirts': 2, 'Village': 1}
        location_tier = tier_map.get(inputs['location'], 1)
        
        # 3. Predict using the 6 features trained in model.py
        input_data = pd.DataFrame([[
            float(inputs['area']), float(inputs['bhk']), float(inputs['age']), 
            float(inputs['bath']), location_tier, float(inputs['quality'])
        ]], columns=['Area_SqFt', 'BHK', 'Age_Years', 'Bathrooms', 'Location_Tier', 'Quality_Score'])

        raw_prediction = model.predict(input_data)[0]
        
        # 4. Logical Floor (₹1800-₹5000/sqft base depending on tier)
        base_rate = float(inputs['area']) * (1800 + (location_tier * 800))
        final_val = max(raw_prediction, base_rate)

        # 5. Advanced Analytics for Dashboard
        trend = "Bullish" if location_tier >= 3 else "Stable"
        quality_impact = "Premium" if float(inputs['quality']) >= 4 else "Standard"
        
        # Comparative Estimates for Table
        comp = {
            'city': format_inr(final_val * (4/location_tier)),
            'village': format_inr(final_val * (1/location_tier))
        }

        return render_template('index.html', 
                               prediction_text=f'₹{format_inr(final_val)}',
                               per_sqft=f'₹{format_inr(final_val / float(inputs['area']))}/sq.ft',
                               location_name=inputs['location'].replace('_', ' '),
                               trend=trend,
                               quality_impact=quality_impact,
                               comp=comp,
                               old_inputs=inputs)
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', prediction_text="Check Data", old_inputs={})

 # Make sure this is at the top of app.py

# ... your other code ...

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)