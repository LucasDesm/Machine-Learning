#%% ---------------------------------------------------------------------------------------------------
from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#%% ---------------------------------------------------------------------------------------------------
# Importer datas
df=pd.read_csv('heart_disease_uci.csv')

#%% ---------------------------------------------------------------------------------------------------
# Traiter dataset
df = df.drop('id', axis=1)
df = df.drop('dataset', axis=1)

label_encoders = {}

str_cols = df.select_dtypes(include=['object','bool']).columns
for col in str_cols:
    label_encoders[col] = LabelEncoder()
    label_encoders[col].fit(df[col].astype(str))
    df[col] = label_encoders[col].transform(df[col].astype(str))


X = df.drop('num',axis=1)
Y=df['num']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()
#on separe les collones par types 

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])
#num = moyenne puis standardise, pas de standar sur les NaN
categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))])
#cat = mode puis one hote encodeur, converti en entier, handle = ignore --> pour les inconnues dans le test sans fail
preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])
#applique chaque sous-pipe aux colonnes corespondante en parallele

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=123)

#%% ---------------------------------------------------------------------------------------------------
# Entrainement
pipe_rf = ImbPipeline(steps=[('preprocessing', preprocessor),('smote', SMOTE(random_state=42)),('model', RandomForestClassifier(random_state=42))])

# Optimisation GridSearch
param_grid_rf = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 15, 20, None],
    'model__min_samples_leaf': [1, 2, 3]
}

grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5)
grid_rf.fit(X_train, Y_train)

Y_pred = grid_rf.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)
print(acc)

score = f1_score(Y_test, Y_pred, average='weighted')

print("Meilleurs paramètres :", grid_rf.best_params_)
print("Meilleur CV score :", grid_rf.best_score_)

print("Precision :", precision_score(Y_test, Y_pred, average='weighted'))
print("Recall :", recall_score(Y_test, Y_pred, average='weighted'))
print("F1-Score :", score)

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
    Y_pred = grid_rf.predict(input_data)
    print(Y_pred)
    result = Y_pred[0]

    return render_template("traitement.html", input_data = input_data, result = result, f1score=f'{score:.2f}')

#%% ---------------------------------------------------------------------------------------------------
# Lance serveur     flask
if __name__ == "__main__":
  app.run(host='0.0.0.0', port=5005, debug=True, use_reloader=False)