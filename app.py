from flask import Flask, render_template, request, jsonify
import joblib
from flask import Flask, render_template, request, jsonify
import joblib
import secrets

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load("iris_model.pkl")

# Secret key for generating and verifying tokens
SECRET_KEY = "ab_123"

# Generate a token for authentication
def generate_token():
    return secrets.token_hex(16)

# Authenticate users with a token
def authenticate_token(token):
    return token == SECRET_KEY


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the token from the request headers
        token = request.headers.get('Authorization')

        # Authenticate the token against your secret key
        if not authenticate_token(token):
            return jsonify({'error': 'Authentication failed'}), 401

        # Get the input data from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction using the loaded model
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        # Determine the class of the predicted flower
        classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = classes[int(prediction)]

        return jsonify({'prediction': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)

