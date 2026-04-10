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

pipe_final = ImbPipeline(steps=[('preprocessing', preprocessor),('smote', SMOTE(random_state=42)),('model', DecisionTreeClassifier())])

pipe_final.fit(X_train, y_train)
y_pred = pipe_final.predict(X_test)

print("\nRésultats (Pipeline + SMOTE) :")
print(f"Precision : {precision_score(y_test, y_pred, average='weighted'):.4f}")
print(f"Recall : {recall_score(y_test, y_pred, average='weighted'):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred, average='weighted'):.4f}")

param_grid = {'model__min_samples_leaf': np.arange(1, 4),'model__max_depth': [10, 15, 20]}

grid = GridSearchCV(pipe_final, param_grid, cv=5)
grid.fit(X_train, y_train)

print(f"Meilleurs paramètres : {grid.best_params_}")
print(f"Meilleur CV score : {grid.best_score_:.4f}")

pipe_optimal = grid.best_estimator_
y_pred_opt = pipe_optimal.predict(X_test)

print("\nRésultats (Pipeline + SMOTE + GridSearchCV) :")
print(f"Precision : {precision_score(y_test, y_pred_opt, average='weighted'):.4f}")
print(f"Recall : {recall_score(y_test, y_pred_opt, average='weighted'):.4f}")
print(f"F1-Score : {f1_score(y_test, y_pred_opt, average='weighted'):.4f}")