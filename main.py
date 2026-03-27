# Projet - Machine learning 2026
#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("heart_disease_uci.csv")
# %%
print(df.info())
print(df.describe())
print(df.isna().sum())

X = df.drop(["num"], axis=1)
Y = df["num"]

# %%
