# Flask app setup
app = Flask(_name_)

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

if _name_ == '_main_':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)