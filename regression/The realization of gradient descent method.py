import numpy as np
import matplotlib.pyplot as plt

k=0 #Initialization slope
b=0 #Initialize intercept
learning_rate=0.0001 #learning rate will affect the speed of machine learning fitting process
epochs =100 #set maximum iterations

#defining cost functions(base on least square method)
def cost_fuc(k,b,x_data,y_data):
    error=0
    for i in range(0,len(x_data)):
        error+=(y_data[i]-(k*x_data[i]+b))**2
        return error / float(len(x_data))

#define gradient_descent functions
def gradient_descent(x_data,y_data,learning_rate,epochs,k,l):
    total=float(len(x_data))
    for i in range(epochs):
        b_grad =0 
        k_grad =0
        for j in range(0,len(x_data)):
            b_grad += (1/total)*((k*x_data[j]+b)-y_data[j])
            k_grad += (1/total)*x_data[j]*((k*x_data[j]+b)-y_data[j]
         b=b-(learning_rate*b_grad)
         k=k-(learning_rate*k_grad)
return k,b
plt.plot(x_data,y_data,'b.')
plt.plot(x_data,k*x_data +b,'r')
plt.show()  


