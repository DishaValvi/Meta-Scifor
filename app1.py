from flask import Flask, render_template, jsonify
import os

app = Flask(__name__)

# Dummy evaluation function for example
def evaluate_model(model, X_train, y_train, X_test, y_test):
    return {"Accuracy": "95%", "F1-Score": "0.93"}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/train')
def train():
    results = {
        'RandomForest': evaluate_model(None, None, None, None, None)
    }
    # Example: Adding XGBoost results
    results['XGBoost'] = evaluate_model(None, None, None, None, None)
    return jsonify(results)

if __name__ == '__main__':  # FIXED __name__
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
