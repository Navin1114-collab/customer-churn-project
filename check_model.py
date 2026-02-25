import joblib
from sklearn.metrics import roc_auc_score, classification_report

X_test = joblib.load(r'C:\Users\navin\customer-churn-project\models\X_test_processed.pkl')
y_test = joblib.load(r'C:\Users\navin\customer-churn-project\models\y_test.pkl')
model = joblib.load(r'C:\Users\navin\customer-churn-project\models\churn_model.pkl')

preds = model.predict_proba(X_test)[:,1]
print('ROC-AUC:', round(roc_auc_score(y_test, preds), 4))
print(classification_report(y_test, model.predict(X_test)))