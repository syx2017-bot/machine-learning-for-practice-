import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

data = np.genfromtxt("data.csv",delimiter=",")
x_data = data[:,1]
y_data = data[:,2]

x_data = data[:,1,np.newaxis]
y_data = data[:,2,np.newaxis]

ploy_regre = PolynomialFeatures(degree=1)
x_ploy=ploy_reg.fit_transform(x_data)
model = LinearRegression()
model.fit(x_ploy,y_data)
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_ploy),c='r')
plt.show()
