from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

model_path = 'model.pkl'
model = None
if os.path.exists(model_path):
    model = pickle.load(open(model_path, 'rb'))

def format_indian_currency(amount):
    """Formats number to Indian numbering system (Lakhs/Crores)"""
    s = str(int(amount))
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
        return render_template('index.html', prediction_text="Error: Model file missing.")
    
    try:
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([np.array(features)])
        
        # Format the result in Indian Rupees
        formatted_price = format_indian_currency(prediction[0])

        return render_template('index.html', 
                               prediction_text=f'Estimated Value: ₹{formatted_price}')
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Check your inputs.")

if __name__ == "__main__":
    app.run(debug=True)