import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt 

data = genfromtxt("data.csv",delimiter=',')
x_data = data[:,:-1]
y_data = data[:,-1]
model = linear_model.LinearRegression()
model.fit(x_data, y_data)
print(model.coef_)
print(model.intercept_)
x_test = [[x,y]] # chose any data in the dataset to test
predict = model.predict(x_test)
print(predict)
