from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv",delimiter=",")
x_data = data[:,0] #extract the column of X
y_data = data[:,1] #extract the column of Y
x_data=data[:,0,np.newaxis]
y_data=data[:,0,np.newaxis]
model = LinearRegression()
model.fit(x_data,y_data)
print(model.coef_,model.intercept_)
model.score(x_data,y_data)
