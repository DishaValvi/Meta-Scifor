import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify, request, render_template_string
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

RANDOM_STATE = 42

data_path_xlsx = 'loan_prediction.csv.xlsx'
if os.path.exists(data_path_xlsx):
    df = pd.read_excel(data_path_xlsx)

if 'Loan_ID' in df.columns:
    df = df.drop(columns=['Loan_ID'])
if 'Loan_Status' in df.columns:
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

df = df.dropna(subset=['Loan_Status'])

for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype(str)

X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='most_frequent')

cat_pipeline = Pipeline(steps=[('imputer', cat_imputer), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
num_pipeline = Pipeline(steps=[('imputer', num_imputer), ('scaler', StandardScaler())])

preprocessor = ColumnTransformer(transformers=[
    ('num', num_pipeline, numeric_features),
    ('cat', cat_pipeline, cat_features)
])

rf_pipeline = Pipeline(steps=[('preproc', preprocessor), ('clf', RandomForestClassifier(random_state=RANDOM_STATE))])
if XGBClassifier is not None:
    xgb_pipeline = Pipeline(steps=[('preproc', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE))])
else:
    xgb_pipeline = rf_pipeline

def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except:
        y_proba = None
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
rf_pipeline.fit(X_train, y_train)
xgb_pipeline.fit(X_train, y_train)
app = Flask(__name__)
html_form = """
<h2>Loan Approval Prediction</h2>
<form action="/predict" method="post">

    <!-- Gender -->
    <label>Gender:</label><br>
    <input type="radio" name="Gender" value="Male" required> Male
    <input type="radio" name="Gender" value="Female"> Female
    <br><br>

    <!-- Married -->
    <label>Married:</label><br>
    <input type="radio" name="Married" value="Yes" required> Yes
    <input type="radio" name="Married" value="No"> No
    <br><br>

    <!-- Dependents -->
    <label>Dependents:</label>
    <select name="Dependents" required>
        <option value="0">0</option>
        <option value="1">1</option>
        <option value="2">2</option>
        <option value="3+">3+</option>
    </select>
    <br><br>

    <!-- Education -->
    <label>Education:</label><br>
    <input type="radio" name="Education" value="Graduate" required> Graduate
    <input type="radio" name="Education" value="Not Graduate"> Not Graduate
    <br><br>

    <!-- Self Employed -->
    <label>Self Employed:</label><br>
    <input type="radio" name="Self_Employed" value="Yes" required> Yes
    <input type="radio" name="Self_Employed" value="No"> No
    <br><br>

    <!-- Applicant Income -->
    <label>Applicant Income:</label>
    <input type="number" name="ApplicantIncome" min="0" required>
    <br><br>

    <!-- Coapplicant Income -->
    <label>Coapplicant Income:</label>
    <input type="number" name="CoapplicantIncome" min="0" required>
    <br><br>

    <!-- Loan Amount -->
    <label>Loan Amount:</label>
    <input type="number" name="LoanAmount" min="0" required>
    <br><br>

    <!-- Loan Amount Term -->
    <label>Loan Amount Term (days):</label>
    <input type="number" name="Loan_Amount_Term" min="0" required>
    <br><br>

    <!-- Credit History -->
    <label>Credit History:</label>
    <select name="Credit_History" required>
        <option value="1.0">Good (1)</option>
        <option value="0.0">Bad (0)</option>
    </select>
    <br><br>

    <!-- Property Area -->
    <label>Property Area:</label>
    <select name="Property_Area" required>
        <option value="Urban">Urban</option>
        <option value="Semiurban">Semiurban</option>
        <option value="Rural">Rural</option>
    </select>
    <br><br>

    <input type="submit" value="Check Approval">
</form>
"""
@app.route('/')
def home():
    return render_template_string(html_form)

@app.route('/train')
def train():
    results = {
        'RandomForest': evaluate_model(rf_pipeline, X_train, y_train, X_test, y_test)
    }
    if XGBClassifier is not None:
        results['XGBoost'] = evaluate_model(xgb_pipeline, X_train, y_train, X_test, y_test)
    return jsonify(results)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = pd.DataFrame([{
        "Gender": request.form["Gender"],
        "Married": request.form["Married"],
        "Dependents": request.form["Dependents"],
        "Education": request.form["Education"],
        "Self_Employed": request.form["Self_Employed"],
        "ApplicantIncome": float(request.form["ApplicantIncome"]),
        "CoapplicantIncome": float(request.form["CoapplicantIncome"]),
        "LoanAmount": float(request.form["LoanAmount"]),
        "Loan_Amount_Term": float(request.form["Loan_Amount_Term"]),
        "Credit_History": float(request.form["Credit_History"]),
        "Property_Area": request.form["Property_Area"]
    }])

    # Ensure string columns stay as strings
    for col in input_data.select_dtypes(include=['object']).columns:
        input_data[col] = input_data[col].astype(str)

    # Predict using the real model
    prediction = rf_pipeline.predict(input_data)[0]
    result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    return f"<h2>{result}</h2>"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
