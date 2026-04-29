#%% ---------------------------------------------------------------------------------------------------
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

#%% ---------------------------------------------------------------------------------------------------
# Importer datas
df=pd.read_csv('heart_disease_uci.csv')

#%% ---------------------------------------------------------------------------------------------------
# Traiter dataset
df = df.drop('id', axis=1)

df['chol'] = df['chol'].replace(0, np.nan)
df['trestbps'] = df['trestbps'].replace(0, np.nan)
df.loc[df['oldpeak'] < 0, 'oldpeak'] = np.nan

colonnes_num = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in colonnes_num : 
    df[col] = df[col].fillna(df[col].mean())
    
colonnes_cat = ['sex','dataset','cp','fbs', 'restecg', 'exang', 'slope', 'thal']

for col in colonnes_cat :
    df[col] = df[col].fillna(df[col].mode()[0])

label_encoders = {}

str_cols = df.select_dtypes(include=['object','bool']).columns
for col in str_cols:
    label_encoders[col] = LabelEncoder()
    label_encoders[col].fit(df[col].astype(str))
    df[col] = label_encoders[col].transform(df[col].astype(str))


X = df.drop('num',axis=1)
Y=df['num']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=0.25, random_state=123)

#%% ---------------------------------------------------------------------------------------------------
# Entrainement
smote = SMOTE(random_state=42)
X_train_res, Y_train_res = smote.fit_resample(X_train, Y_train)

steps = [('scaler',StandardScaler()),('model', DecisionTreeClassifier())]

p1 = Pipeline(steps)
p1.fit(X_train_res, Y_train_res)
print(p1.score(X_test, Y_test))

#%% ---------------------------------------------------------------------------------------------------
# Créer app flask avec 2 routes
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
        'dataset': ['Cleveland'],
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
    input_data = input_data.astype({
        'age': int,
        'trestbps': int,
        'chol': int,
        'thalch': int,
        'oldpeak': float,
        'ca': int,
    })
    input_data_str = input_data.select_dtypes(include=['object','bool']).columns
    for col in input_data_str:
        if col in label_encoders:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
    print(input_data)
    Y_pred = p1.predict(input_data)
    print(Y_pred)
    result = Y_pred[0]

    return render_template("traitement.html", result = result)
    return render_template("traitement.html", input_data = input_data, result = result)

#%% ---------------------------------------------------------------------------------------------------
# Lance serveur flask
if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5005, debug=True, use_reloader=False)