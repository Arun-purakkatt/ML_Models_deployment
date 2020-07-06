from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import pickle
import numpy as np



app= Flask(__name__)

# Load the Random Forest CLassifier model
filename = 'insurance-prediction-rfr-model.pkl'
model= pickle.load(open(filename, 'rb'))


cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        sex = str(request.form['sex'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = str(request.form['smoker'])
        region = str(request.form['region'])

        data_unseen = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(data_unseen)
        return render_template('home.html', pred='Expected Bill will be {}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)
