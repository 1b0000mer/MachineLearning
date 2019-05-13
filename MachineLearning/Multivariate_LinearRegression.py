import numpy as np
from func_LR import computeCost, gradientDescent
import os           #for getting file path
import pandas as pd
from matplotlib import pyplot as plt

filePath = os.getcwd() + '\multiLR\data2.txt'
feature = ['Size', 'Room', 'Price']
training_data = pd.read_csv(filePath, names=feature)

print(training_data.shape)
print(training_data.head())
print('#####################')
#training_data.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.pyplot.show()

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

theta_final, J_cost = gradientDescent(X, y, theta, m, loop=1000)
print(theta_final)
print(J_cost[-1]) #min

#guess
size = 1100.0
room = 3.0
test = ([size, room, 1.] * theta_final)
print('guest for: size=%d, room=%d' %(size, room))
print(test.sum(axis = 1))
