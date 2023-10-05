from flask import Flask, render_template, request
import joblib
import sklearn


app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load("iris_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])

        # Make a prediction using the loaded model
        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]

        # Determine the class of the predicted flower
        classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = classes[int(prediction)]

        return render_template('index.html', prediction=predicted_class)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
