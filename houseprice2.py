import pandas as pd
from sklearn.linear_model import LinearRegression

def house_price(num):


    df = pd.read_csv('D:/jupyter/datasets/homeprices.csv')
    model = LinearRegression()
    model.fit(df[['area']], df.price)

    prediction = model.predict([[num]])

    return prediction[0].round(2)


print(house_price(3300))
