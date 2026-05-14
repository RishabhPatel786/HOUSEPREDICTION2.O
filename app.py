from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# 1. Load the model and the column structure
# This assumes you saved a dictionary containing 'model' and 'columns' in model_data.pkl
model_path = 'model_data.pkl'
model = None
model_columns = []

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        model_columns = data['columns']

def format_indian_currency(amount):
    """
    Formats number to Indian numbering system (Lakhs/Crores)
    and prevents negative or zero values.
    """
    amount = max(50000, round(amount)) # Minimum floor price of 50k for realism
    s = str(amount)
    if len(s) <= 3: return s
    last_three = s[-3:]
    remaining = s[:-3]
    # Indian comma placement logic
    remaining = ",".join([remaining[max(i-2, 0):i] for i in range(len(remaining), 0, -2)][::-1])
    return f"{remaining},{last_three}"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model data not found on server.")
    
    try:
        # 1. Capture Form Data
        area = float(request.form.get('area'))
        bhk = float(request.form.get('bhk'))
        age = float(request.form.get('age'))
        bath = float(request.form.get('bath'))
        location = request.form.get('location')

        # 2. Reconstruct the Feature Vector
        # We must create a DataFrame with all zeros for all encoded columns
        input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
        
        # 3. Map numerical inputs to the correct columns
        # Ensure these names match the column names in your housing_data.csv exactly
        input_df['Area_SqFt'] = area
        input_df['BHK'] = bhk
        input_df['Age_Years'] = age
        input_df['Bathrooms'] = bath
        
        # 4. Handle the One-Hot Encoded Location
        # If user selected 'City', we set 'Location_City' to 1
        loc_col = 'Location_' + location
        if loc_col in model_columns:
            input_df[loc_col] = 1

        # 5. Execute Prediction
        prediction = model.predict(input_df)
        
        # 6. Format and Return
        formatted_price = format_indian_currency(prediction[0])

        return render_template('index.html', 
                               prediction_text=f'₹{formatted_price}',
                               area=area, 
                               bhk=int(bhk))

    except Exception as e:
        print(f"Prediction Error: {e}")
        return render_template('index.html', prediction_text="Error: Could not calculate price. Check inputs.")

if __name__ == "__main__":
    # Use environment port for Render or default to 5000 for local
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)