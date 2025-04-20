from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# model and preprocessor
model = joblib.load('models/churn_model.pkl')  # Path to model in models/ folder
preprocessor = joblib.load('models/preprocessor.pkl')  # Path to preprocessor

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    df = pd.DataFrame([data])
    
    processed_data = preprocessor.transform(df)
    
    churn_prob = model.predict_proba(processed_data)[0][1]  # Probability of class "1" (Churn)
    
    return jsonify({'churn_probability': float(churn_prob)})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
