import numpy as np


def gradient_descent(x,y):       # y = mx + b, objective to get the value of m and b where m is the gradient and b is the y-intercept
    m_curr = b_curr = 0          #start with some value of m_current and b_current
    iterations = 2000              # define the number of iterations
    n = len(x)                   # lenght of array x considering that x and y are of the same length else throw an error
    learning_rate = 0.08         # define the step size of the descent

    for i in range(iterations):                 # number of loops 
        y_predicted = m_curr * x + b_curr       # y = mx + b
        # cost1 = (sum((y - y_predicted)**2))/n    #same as cost equation
        cost = (1/n)*sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))      #derivative of m 
        bd = -(2/n) * sum(y-y_predicted)        #derivative of b
        m_curr = m_curr - learning_rate * md    # m = m - learning rate * derivative of m
        b_curr = b_curr - learning_rate * bd    # b = b - learnign rate * derivative of b

        print(f'm: {m_curr} b: {b_curr} cost: {cost} iterations: {i}')
        # print(f'cost is: {cost1}')
        

        
    

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)