from diagrams import Diagram
from diagrams.onprem.client import User
from diagrams.programming.language import Python
from diagrams.onprem.compute import Server
from diagrams.generic.storage import Storage
import os

os.environ["PATH"] += r";C:\Program Files\Graphviz\bin"
os.chdir(r"C:\Users\navin\customer-churn-project\images")

with Diagram("Telco Customer Churn Predictor", filename="architecture", outformat="png", show=False, direction="LR"):
    data = Storage("IBM Telco Dataset\n7,043 customers")
    preprocess = Python("Preprocessing\nStandardScaler\nOneHotEncoder")
    model = Python("XGBoost Model\nROC-AUC: 0.8059")
    api = Server("Flask REST API\n/predict endpoint")
    dashboard = User("Streamlit\nDashboard")

    data >> preprocess >> model >> api >> dashboard