import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
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


