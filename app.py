from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor
model = joblib.load('models/churn_model.pkl')  # Path to model in models/ folder
preprocessor = joblib.load('models/preprocessor.pkl')  # Path to preprocessor

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from request
    data = request.json
    
    # Convert data to DataFrame
    df = pd.DataFrame([data])
    
    # Preprocess the data
    processed_data = preprocessor.transform(df)
    
    # Predict churn probability
    churn_prob = model.predict_proba(processed_data)[0][1]  # Probability of class "1" (Churn)
    
    # Return result as JSON
    return jsonify({'churn_probability': float(churn_prob)})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app