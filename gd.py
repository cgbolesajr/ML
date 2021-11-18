import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import math


def predict_using_sklearn():
    # df = pd.read_csv("d:/jupyter/datasets/test_scores.csv")  
    r = LinearRegression()
    r.fit(df[['math']], df.cs)
    return r.coef_, r.intercept_


def gradient_descent(x, y):
    m_curr = 0
    b_curr = 0
    iterations = 100000
    n = len(x)
    learning_rate = 0.0002

    cost_previous = 0

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr   # m*x + b
        # cost = (1/n)*sum([value**2 for value in (y-y_predicted)])
        cost = (1/n)*sum((y-y_predicted)**2)
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        if math.isclose(cost, cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost
        print(f"m {m_curr}, b {b_curr}, cost {cost}, iteration {i}")

    return m_curr, b_curr


if __name__ == "__main__":
    df = pd.read_csv("d:/jupyter/datasets/test_scores.csv")
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x, y)
    print(f"Using gradient descent function: Coef {m} Intercept {b}")

    m_sklearn, b_sklearn = predict_using_sklearn()
    print(f"Using sklearn: Coef {m_sklearn[0]} Intercept {b_sklearn}")

# @100k iterations and .0002 learning rate

# Using gradient descent function: Coef 1.0204362110820604 Intercept 1.72387921708579
# Using sklearn: Coef 1.017736237856933 Intercept 1.9152193111568891
