from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained regression model
model = joblib.load('taxi_fare_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features
        features = [
        float(request.form['trip_duration']),        
        float(request.form['distance_traveled']),    
        float(request.form['num_of_passengers']),
        int(request.form['surge_applied']),
        float(request.form['KPH']),
        float(request.form['tip']),
        float(request.form['trip_duration']) * float(request.form['distance_traveled']), 
        float(request.form['KPH']) * int(request.form['surge_applied']),
        float(request.form['tip']) * float(request.form['num_of_passengers'])
    ]

        prediction = model.predict([features])[0]
        return render_template('index.html', prediction_text=f'Estimated Fare: ${prediction:.2f}')
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)