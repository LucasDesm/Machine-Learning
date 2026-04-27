import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE

df=pd.read_csv('C:/ML/Machine Learning/Projet finale/Machine-Learning/heart_disease_uci.csv')
print(df.shape)

#traitement des données basiques
df = df.drop('id', axis=1)
#Valeur aberrante
df['chol'] = df['chol'].replace(0, np.nan)
df['trestbps'] = df['trestbps'].replace(0, np.nan)
df.loc[df['oldpeak'] < 0, 'oldpeak'] = np.nan

print('Valeurs manquantes:', df.isnull().sum().sum())

colonnes_num = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
for col in colonnes_num : 
    df[col] = df[col].fillna(df[col].mean())
    
colonnes_cat = ['sex','dataset','cp','fbs', 'restecg', 'exang', 'slope', 'thal']

for col in colonnes_cat :
    df[col] = df[col].fillna(df[col].mode()[0])

def categorical_features(df):
    object_columns = df.select_dtypes(include=['object','bool']).columns
    label_encoder = LabelEncoder()
    for col in object_columns:
        df[col] = label_encoder.fit_transform(df[col].astype(str))
    return df

df = categorical_features(df)

X = df.drop('num',axis=1)
y=df['num']

print(np.mean(X))
print(np.std(X))
print(df.info())

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.25, random_state=123)

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import  StandardScaler

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

steps = [('scaler',StandardScaler()),('model', DecisionTreeClassifier())]

p1 = Pipeline(steps)

p1.fit(X_train, y_train)

y_pred = p1.predict(X_test)

from sklearn.metrics import recall_score, precision_score, f1_score

print("Precision :", precision_score(y_test, y_pred, average='weighted'))
print("Recall :", recall_score(y_test, y_pred, average='weighted'))
print("F1-Score :", f1_score(y_test, y_pred, average='weighted'))

#standardisation

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

#comparaison des modéles
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'AdaBoost': AdaBoostClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()
}

results = []

for name, model in models.items():
    model.fit(X_train_scaled, y_train_res)
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({'Modèle': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1-Score': f1})
    print(name, round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4))
    
best = max(results, key=lambda x: x['F1-Score'])
print(best)


#optimisations avec GridSearch 
#Random Forest
param_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_leaf': [1, 2, 3]}

grid_rf = GridSearchCV(RandomForestClassifier(), param_rf, cv=5, scoring='f1_weighted')
grid_rf.fit(X_train_scaled, y_train_res)
print('Meilleurs paramètres :', grid_rf.best_params_)
print('CV Score :', grid_rf.best_score_)
print('Test F1 :', f1_score(y_test, grid_rf.predict(X_test_scaled), average='weighted'))

#KNN
param_knn = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_knn, cv=5, scoring='f1_weighted')
grid_knn.fit(X_train_scaled, y_train_res)
print('Meilleurs paramètres :', grid_knn.best_params_)
print('CV Score :' ,grid_knn.best_score_)
print('Test F1 :', f1_score(y_test, grid_knn.predict(X_test_scaled), average='weighted'))

# SVM
param_svm = {'C': [0.1, 1, 10],'kernel': ['rbf', 'linear']}
grid_svm = GridSearchCV(SVC(), param_svm, cv=5, scoring='f1_weighted')
grid_svm.fit(X_train_scaled, y_train_res)
print('Meilleurs paramètres :', grid_svm.best_params_)
print('CV Score :', grid_svm.best_score_)
print('Test F1 : ', f1_score(y_test, grid_svm.predict(X_test_scaled), average='weighted'))

#Gradient Boostin

param_gb = {'n_estimators': [50, 100],'max_depth': [3, 5],'learning_rate': [0.1, 0.2]}
grid_gb = GridSearchCV(GradientBoostingClassifier(), param_gb, cv=5, scoring='f1_weighted')
grid_gb.fit(X_train_scaled, y_train_res)
print('Meilleurs paramètres :', grid_gb.best_params_)
print('CV Score : ', grid_gb.best_score_)
print('Test F1 :', f1_score(y_test, grid_gb.predict(X_test_scaled), average='weighted'))




