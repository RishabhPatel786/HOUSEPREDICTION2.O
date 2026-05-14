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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', prediction_text="Error: Model file not found on server.")
    
    try:
        # Get data from form and convert to float
        features = [float(x) for x in request.form.values()]
        final_features = [np.array(features)]
        
        prediction = model.predict(final_features)
        # Format as currency: $1,234,567.89
        output = "{:,.2f}".format(prediction[0])

        return render_template('index.html', prediction_text=f'Estimated Value: ${output}')
    except Exception as e:
        return render_template('index.html', prediction_text="Error: Please check your input values.")

if __name__ == "__main__":
    app.run(debug=True)