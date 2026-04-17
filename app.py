# Install Flask if not already installed
!pip install Flask

from flask import Flask, request, jsonify
import joblib
import pandas as pd
app = Flask(__name__)

# Define the filename for the model
model_filename = 'logistic_regression_diabetes_model.joblib'

# Save the model to disk
joblib.dump(model, model_filename)

print(f"Model successfully exported to '{model_filename}'")

# You can verify by loading it back (optional)
# loaded_model = joblib.load(model_filename)
# print("\nModel loaded successfully.")

# Load the trained model and the scaler
model = joblib.load('logistic_regression_diabetes_model.joblib')
scaler = joblib.load('scaler.joblib') # Load the scaler

# Initialize Flask app


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()

        # Convert input data to DataFrame, ensuring correct order of features
        # The order of features should match the order used during training
        # Based on df.columns[:-1] from initial training
        features = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        input_df = pd.DataFrame([data], columns=features)

        # Scale the input data using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0].tolist()

        # Return prediction
        return jsonify({
            'prediction': int(prediction),
            'prediction_probability': prediction_proba
        })

# To run the Flask app (this part typically runs in a separate script or environment)
if __name__ == '__main__':
    app.run(debug=True) 
