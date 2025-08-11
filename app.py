import os
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, jsonify
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

RANDOM_STATE = 42

# Load data
data_path_xlsx = 'loan_prediction.csv.xlsx'
if os.path.exists(data_path_xlsx):
    df = pd.read_excel(data_path_xlsx)

# Drop ID and map target
if 'Loan_ID' in df.columns:
    df = df.drop(columns=['Loan_ID'])
if 'Loan_Status' in df.columns:
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# Drop rows where target is NaN
df = df.dropna(subset=['Loan_Status'])

# Ensure all categorical columns are strings
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

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return "Loan Prediction Model API is running."

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
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    try:
        new_data = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Failed to read file: {e}'}), 400

    # Ensure same preprocessing as training
    for col in new_data.select_dtypes(include=['object']).columns:
        new_data[col] = new_data[col].astype(str)

    # Predict using trained RF model
    rf_pipeline.fit(X_train, y_train)
    predictions = rf_pipeline.predict(new_data)
    return jsonify({'predictions': predictions.tolist()})
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
