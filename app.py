from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the saved model with a safety check
model_path = 'model.pkl'
model = None
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))

def format_indian_currency(amount):
    """Formats number to Indian numbering system (e.g., 1,00,000)"""
    amount = round(amount)
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
        return render_template('index.html', prediction_text="Model file missing. Please run model.py locally.")
    
    try:
        # Extracting property-specific features
        area = float(request.form.get('area'))
        bhk = float(request.form.get('bhk'))
        age = float(request.form.get('age'))
        bath = float(request.form.get('bath'))
        
        # Prepare for prediction (must match order in model.py)
        input_data = np.array([[area, bhk, age, bath]])
        prediction = model.predict(input_data)
        
        # Format the result
        formatted_price = format_indian_currency(prediction[0])

        return render_template('index.html', 
                               prediction_text=f'₹{formatted_price}',
                               area=area, bhk=int(bhk))
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Invalid input data.")

if __name__ == "__main__":
    app.run(debug=True)