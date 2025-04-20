from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import pandas as pd

# Load processed data (update paths if needed)
X_train_processed = joblib.load('../models/X_train_processed.pkl')
y_train = joblib.load('../models/y_train.pkl')

# Train model (handle class imbalance)
model = XGBClassifier(
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # Adjust for imbalance
    random_state=42
)
model.fit(X_train_processed, y_train)

# Save model
joblib.dump(model, '../models/churn_model.pkl')
print("Model trained and saved!")