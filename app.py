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
        return render_template('index.html', prediction_text="Error: Model missing.")

    try:
        # 1. Capture Inputs
        area = float(request.form.get('area'))
        bhk = float(request.form.get('bhk'))
        age = float(request.form.get('age'))
        bath = float(request.form.get('bath'))
        quality = float(request.form.get('quality')) # Range 1-5
        location = request.form.get('location')

        # 2. Map Location to Numeric Tier (Feature Engineering)
        # Higher tier = Higher base price multiplier
        tier_map = {'City': 4, 'Main_Road': 3, 'Outskirts': 2, 'Village': 1}
        location_tier = tier_map.get(location, 1)

        # 3. Create DataFrame matching model columns
        # Note: Your model.py should be trained on: Area_SqFt, BHK, Age_Years, Bathrooms, Location_Tier, Quality_Score
        input_data = pd.DataFrame([[area, bhk, age, bath, location_tier, quality]], 
                                  columns=['Area_SqFt', 'BHK', 'Age_Years', 'Bathrooms', 'Location_Tier', 'Quality_Score'])

        # 4. Predict
        prediction = model.predict(input_data)[0]
        
        # 5. Logical Floor: Ensures price is never unrealistically low for a City
        base_rate = area * (1000 * location_tier) # Minimum ₹1000/sqft for Village, ₹4000 for City
        final_val = max(prediction, base_rate)

        return render_template('index.html', 
                               prediction_text=f'₹{format_inr(final_val)}',
                               per_sqft=f'₹{format_inr(final_val/area)}/sq.ft',
                               location_name=location.replace('_', ' '),
                               area=area, bhk=int(bhk))
    except Exception as e:
        return render_template('index.html', prediction_text="Calculation Error")

if __name__ == "__main__":
    app.run(debug=True)