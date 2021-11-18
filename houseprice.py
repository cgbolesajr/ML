from sklearn.linear_model import LinearRegression
import pandas as pd


def house_price(d):
    #training
    df = pd.read_csv('D:/jupyter/datasets/homeprices.csv')
    model = LinearRegression()
    model.fit(df[['area']], df.price)

    #predicting prices
    prediction = model.predict(d)
    d['prices'] = prediction.round(2)
    p = d.to_csv('predictions.csv', index=False)
    print(f'coef: {model.coef_} intercept: {model.intercept_}')
    
    return d['prices']



d = pd.read_csv('D:/jupyter/datasets/areas.csv')
# print(d)

print(house_price(d))


