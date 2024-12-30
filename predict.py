import numpy as np
import pickle

from flask import Flask, jsonify, request
    
app = Flask(__name__)

model_path = './model.bin'
with open(model_path, 'rb') as f:
    dv, model = pickle.load(f)

@app.route('/predict', methods=["POST"])
def predict():
    features = request.get_json()

    X = dv.transform(features)
    y_pred = model.predict(X)[0]
    # Convert the log1p back to the original value
    y_pred = np.expm1(y_pred)

    result = {
        'predicted_salary': float(y_pred),
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)