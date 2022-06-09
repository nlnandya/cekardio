from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('models/model.pkl', 'rb'))
scaler_ = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/mulai-prediksi")
def start_predict():
    return render_template("start-predict.html")


@app.route("/hasil-prediksi", methods=['POST'])
def predict_result():
    age = int(request.form['age'])
    gender = float(request.form['gender'])
    height = int(request.form["height"])
    weight = int(request.form["weight"])
    systolic = int(request.form["systolic"])
    diastolic = int(request.form["diastolic"])
    cholesterol = int(request.form["cholesterol"])
    glucose = int(request.form["glucose"])
    smoke = int(request.form["smoke"])
    alcohol = int(request.form["alcohol"])
    active = int(request.form["active"])
    bmi = float(request.form["bmi"])

    val = [age, gender, height, weight, systolic, diastolic,
           cholesterol, glucose, smoke, alcohol, active, bmi]
    val = scaler_.transform([val])
    print("Input Values:", val)
    val_predict = model.predict(val)
    return render_template('predict-result.html', data=val_predict)


if __name__ == "__main__":
    app.run(debug=True)
