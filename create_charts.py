import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load(r'C:\Users\navin\customer-churn-project\models\churn_model.pkl')

feature_names = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'Contract_One year', 'Contract_Two year',
    'InternetService_Fiber optic', 'InternetService_No',
    'PaymentMethod_Credit card', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

importances = model.feature_importances_

plt.figure(figsize=(10, 6))
colors = ['#FF4B4B' if i == importances.argmax() else '#4B8BFF' for i in range(len(importances))]
bars = plt.barh(feature_names, importances, color=colors)
plt.xlabel('Feature Importance Score', fontsize=12)
plt.title('XGBoost Feature Importance — Customer Churn Prediction', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\navin\customer-churn-project\images\feature_importance.png', dpi=150, bbox_inches='tight')
print("Feature importance chart saved.")

labels = ['No Churn (73%)', 'Churn (27%)']
sizes = [5174, 1869]
colors = ['#4B8BFF', '#FF4B4B']
plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90,
        textprops={'fontsize': 13})
plt.title('Customer Churn Distribution\nIBM Telco Dataset — 7,043 customers', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\navin\customer-churn-project\images\churn_distribution.png', dpi=150, bbox_inches='tight')
print("Churn distribution chart saved.")
