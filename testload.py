import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

df = pd.read_csv('d:/jupyter/datasets/homeprices.csv')
model = LinearRegression()
model.fit(df[['area']], df.price)

# with open('model_pickle', 'rb') as file:
#     pickle.load(file)


# print(model.predict([[5000]]))

mj = joblib.load('model_joblib')
print(f'coef m: {mj.coef_[0]}')
print(f'intercept b: {mj.intercept_}')
print(f'prediction: {mj.predict([[5000]])[0]} pesos only for the house')

