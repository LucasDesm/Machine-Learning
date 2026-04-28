from flask import Flask, render_template, request
import pandas as pd

# Tuto suivi: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-ii-templates

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def traitement():
    donnees = request.form
    age = donnees.get('age')
    sex = donnees.get('sex')
    cp = donnees.get('cp')
    trestbps = donnees.get('trestbps')
    chol = donnees.get('chol')
    fbs = donnees.get('fbs')
    restecg = donnees.get('restecg')
    thalch = donnees.get('thalch')
    exang = donnees.get('exang')
    oldpeak = donnees.get('oldpeak')
    slope = donnees.get('slope')
    ca = donnees.get('ca')
    thal = donnees.get('thal')
    
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalch': [thalch],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })

    result = "ok"
    if result :
        return render_template("traitement.html", input_data = input_data, result = result)
    else:
        return render_template("traitement.html")