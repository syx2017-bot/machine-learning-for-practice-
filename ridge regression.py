import numpy as np
from numpy import genfromtxt
from sklearn import linear_model
import matplotlib.pyplot as plt

data = genfromtxt("data.csv",delimiter=',')
x_data = data[1:,2:]
y_data = data[1:,1]
alphas = np.linspace(0.001, 1)
model = linear_model.RidgeCV(alphas=alphas, store_cv_values=True)
model.fit(x_data, y_data)
print(model.alpha_)
plt.plot(alphas, model.cv_values_.mean(axis=0))
plt.plot(model.alpha_, min(model.cv_values_.mean(axis=0)),'r.')
plt.show()
model.predict(x_data[2,np.newaxis])
