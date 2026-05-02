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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


df=pd.read_csv('C:/ML/Machine Learning/Projet finale/Machine-Learning/heart_disease_uci.csv')
print(df.shape)

X = df.drop('num', axis=1)
y = df['num']

numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

categorical_features = X.select_dtypes(include=['object', 'bool']).columns.tolist()

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='mean')),('scaler', StandardScaler())])

categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),('encoder', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features),('cat', categorical_transformer, categorical_features)])

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

#SVM
pipe_svm = ImbPipeline(steps=[('preprocessing', preprocessor),('smote', SMOTE(random_state=42)),('model', SVC())])

pipe_svm.fit(X_train, y_train)
y_pred_svm = pipe_svm.predict(X_test)

print("Precision :", precision_score(y_test, y_pred_svm, average='weighted'))
print("Recall : ", recall_score(y_test, y_pred_svm, average='weighted'))
print("F1-Score :", f1_score(y_test, y_pred_svm, average='weighted'))

# Optimisation GridSearch
param_grid_svm = {
    'model__C': [0.1, 1, 10],
    'model__kernel': ['rbf', 'linear']
}

grid_svm = GridSearchCV(pipe_svm, param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)

print("Meilleurs paramètres :", grid_svm.best_params_)
print("Meilleur CV score : ", grid_svm.best_score_)

pipe_svm_optimal = grid_svm.best_estimator_
y_pred_svm_opt = pipe_svm_optimal.predict(X_test)

print("Precision :", precision_score(y_test, y_pred_svm_opt, average='weighted'))
print("Recall :", recall_score(y_test, y_pred_svm_opt, average='weighted'))
print("F1-Score :", f1_score(y_test, y_pred_svm_opt, average='weighted'))

#random forest
pipe_rf = ImbPipeline(steps=[('preprocessing', preprocessor),('smote', SMOTE(random_state=42)),('model', RandomForestClassifier(random_state=42))])

pipe_rf.fit(X_train, y_train)
y_pred_rf = pipe_rf.predict(X_test)

print("Precision : ", precision_score(y_test, y_pred_rf, average='weighted'))
print(f"Recall : ", recall_score(y_test, y_pred_rf, average='weighted'))
print(f"F1-Score :", f1_score(y_test, y_pred_rf, average='weighted'))

# Optimisation GridSearch
param_grid_rf = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [10, 15, 20, None],
    'model__min_samples_leaf': [1, 2, 3]
}

grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5)
grid_rf.fit(X_train, y_train)

print("Meilleurs paramètres :", grid_rf.best_params_)
print("Meilleur CV score :", grid_rf.best_score_)

pipe_rf_optimal = grid_rf.best_estimator_
y_pred_rf_opt = pipe_rf_optimal.predict(X_test)

print("Precision :", precision_score(y_test, y_pred_rf_opt, average='weighted'))
print("Recall :", recall_score(y_test, y_pred_rf_opt, average='weighted'))
print("F1-Score :", f1_score(y_test, y_pred_rf_opt, average='weighted'))












