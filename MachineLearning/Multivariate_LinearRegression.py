import os           #for getting file path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from func_LR import computeCost, gradientDescent

###input
filePath = os.getcwd() + '\multiLR\data.csv'
feature = ['Size', 'Room', 'Price']
training_data = pd.read_csv(filePath, names=feature)
#for plot
x_pop = np.linspace(training_data.values[:,0].min(), training_data.values[:,0].max(), 5,dtype=int)
y_pop = np.linspace(training_data.values[:,1].min(), training_data.values[:,1].max(), 5,dtype=int)
######

print(training_data.shape)
print(training_data.head())
print('######################')
#training_data.plot()
#plt.show()

fig1 = plt.figure()
ax = plt.axes(projection='3d')

#Data for three-dimensional scattered points
zdata = training_data.values[:,2]
xdata = training_data.values[:,0]
ydata = training_data.values[:,1]
ax.set_xlabel('Size')
ax.set_ylabel('Room')
ax.set_zlabel('Price')
ax.set_title('Training Data')
ax.scatter3D(xdata, ydata, zdata);

###Standardization
training_data = (training_data - training_data.mean())/training_data.std()

matrix = training_data.values
m,n = matrix.shape
X_col = matrix[:,0:n-1]
X = np.insert(X_col, 0, values=1, axis=1)
y = matrix[:,n-1:n]
#print(X[:5])
#print(y[:5])

theta = np.zeros((1, X.shape[1]))
#print(theta)
#print(X.shape, y.shape, theta.shape)

#print(computeCost(X, y, theta))

###begin
'''
Multiple Regression Model: b0 + b1x1 + b2x2 + bpxp + ... + e    (linear parameters)
b0,b1,..: weigth
x1,x2...: ...feature?
e: error
'''
loop = 2000
theta_final, J_cost = gradientDescent(X, y, theta, m, loop=2000)
#theta_final = gradientDescent(X, y, theta, m, loop=500)
#########

print('Theta:')
print(theta_final)
print('cost func: %f' %J_cost[-1])
#print(J_cost[-1]) #min

###theta prediction fig
#ax.scatter3D(x_pop, y_pop, f, 'r', label='Prediction')
f = ((theta_final[0, 0] * x_pop) + (theta_final[0, 1] * y_pop) + theta_final[0, 2])*100000
ax.plot3D(x_pop, y_pop, f, 'green', label='prediction');
########

###guess
print('######################')
size = 2000.0
room = 3.0
test = ([size, room, 1.] * theta_final)
print('guest for: size=%d, room=%d' %(size, room))
print('price: %f' %(test.sum(axis = 1)*100000))

####cost func fig
fig2 = plt.figure()
cs = plt.axes()
cs.plot(np.arange(loop), J_cost, 'r')
cs.set_xlabel('So Lan Lap')
cs.set_ylabel('Ham Chi Phi')
cs.set_title('Ham Chi Phi vs. So Lan Lap')
plt.show()