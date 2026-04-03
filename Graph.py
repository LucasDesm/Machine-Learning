import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv('C:/ML/Machine Learning/Projet finale/Machine-Learning/heart_disease_uci.csv')
print(df.shape)

#Exploration des données
print(df.head(10))
print(df.columns.tolist())
print(df.dtypes)
print(df.info())

#Quels sont les valeurs manquantes ? 
print(df.isnull().sum())

#Total des valeurs manquante
print(df.isnull().sum().sum())

#Stat descriptive
print(df.describe())

#Variable ciblé 
print(df['num'].unique())
print(df['num'].value_counts())

#Les différents Features
features = ['sex', 'dataset', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']
for feat in features:
    print(df[feat].value_counts())
    
#Graphiques 
#Histogramme
import matplotlib.pyplot as plt

df['id'].hist(by=df['num'], figsize=(12,8))
df.plot(kind='scatter', x='num', y='id' )
plt.show()
plt.plot()

#histogramme
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].hist(df['age'].dropna(), bins=15, color='steelblue', edgecolor='black')
axes[0, 0].set_title('Histogramme de age')
axes[0, 0].set_xlabel('age')
axes[0, 0].set_ylabel('Fréquence')

axes[0, 1].hist(df['trestbps'].dropna(), bins=15, color='steelblue', edgecolor='black')
axes[0, 1].set_title('Histogramme de trestbps')
axes[0, 1].set_xlabel('trestbps')
axes[0, 1].set_ylabel('Fréquence')

axes[0, 2].hist(df['chol'].dropna(), bins=15, color='steelblue', edgecolor='black')
axes[0, 2].set_title('Histogramme de chol')
axes[0, 2].set_xlabel('chol')
axes[0, 2].set_ylabel('Fréquence')

axes[1, 0].hist(df['thalch'].dropna(), bins=15, color='steelblue', edgecolor='black')
axes[1, 0].set_title('Histogramme de thalch')
axes[1, 0].set_xlabel('thalch')
axes[1, 0].set_ylabel('Fréquence')

axes[1, 1].hist(df['oldpeak'].dropna(), bins=15, color='steelblue', edgecolor='black')
axes[1, 1].set_title('Histogramme de oldpeak')
axes[1, 1].set_xlabel('oldpeak')
axes[1, 1].set_ylabel('Fréquence')

axes[1, 2].hist(df['ca'].dropna(), bins=4, color='steelblue', edgecolor='black')
axes[1, 2].set_title('Histogramme de ca')
axes[1, 2].set_xlabel('ca')
axes[1, 2].set_ylabel('Fréquence')

plt.tight_layout()
plt.show()


# boxplots


fig, axes = plt.subplots(2, 3, figsize=(12, 8))

axes[0, 0].boxplot(df['age'].dropna())
axes[0, 0].set_title('Boxplot de age')
axes[0, 0].set_ylabel('age')

axes[0, 1].boxplot(df['trestbps'].dropna())
axes[0, 1].set_title('Boxplot de trestbps')
axes[0, 1].set_ylabel('trestbps')

axes[0, 2].boxplot(df['chol'].dropna())
axes[0, 2].set_title('Boxplot de chol')
axes[0, 2].set_ylabel('chol')

axes[1, 0].boxplot(df['thalch'].dropna())
axes[1, 0].set_title('Boxplot de thalch')
axes[1, 0].set_ylabel('thalch')

axes[1, 1].boxplot(df['oldpeak'].dropna())
axes[1, 1].set_title('Boxplot de oldpeak')
axes[1, 1].set_ylabel('oldpeak')

axes[1, 2].boxplot(df['ca'].dropna())
axes[1, 2].set_title('Boxplot de ca')
axes[1, 2].set_ylabel('ca')

plt.tight_layout()
plt.show()


#scatter plot
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].set_scatter(df['age'],color='blue', alpha=0.5)
axes[0, 0].set_title('age vs thalch')
axes[0, 0].set_xlabel('age',color='blue')
axes[0, 0].set_ylabel('thalch',color='red')

axes[0, 1].scatter(df['age'], df['chol'], alpha=0.5)
axes[0, 1].set_title('age vs chol')
axes[0, 1].set_xlabel('age')
axes[0, 1].set_ylabel('chol')

axes[1, 0].scatter(df['trestbps'], df['chol'], alpha=0.5)
axes[1, 0].set_title('trestbps vs chol')
axes[1, 0].set_xlabel('trestbps')
axes[1, 0].set_ylabel('chol')

axes[1, 1].scatter(df['oldpeak'], df['thalch'], alpha=0.5)
axes[1, 1].set_title('oldpeak vs thalch')
axes[1, 1].set_xlabel('oldpeak')
axes[1, 1].set_ylabel('thalch')

plt.tight_layout()
plt.show()

#matrice de corrélation 
colonnes_num = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca', 'num']
corr = df[colonnes_num].corr()

print("Matrice de corrélation :")
print(corr.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True)
plt.title('Matrice de corrélation')
plt.tight_layout()
plt.show()
