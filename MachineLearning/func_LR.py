import numpy as np
from matplotlib import pyplot as plt

def h_X(X, theta):
    return np.dot(X, theta.T)

def computeCost(X, y, theta, m):
    err = np.power((h_X(X, theta) - y),2)
    J = (1.0/(2*m)) * np.sum(err)

    return J

def gradientDescent(X, y, theta, m, alpha = 0.01, loop = 10):
    theta_center = np.zeros(theta.shape)
    var_Theta = theta.shape[1]
    J_center = np.zeros(loop)
    for i in range(loop):
        err = h_X(X,theta) - y
        for j in range(var_Theta):
            x_ij = np.reshape(X[:,j],(len(X),1))
            term = np.multiply(err,x_ij)
            theta_center[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        theta = theta_center
        J_center[i] = computeCost(X,y,theta,m)
    
    return theta, J_center

#def plotfig():
#    x_pop = np.linspace(X_dacTinh.min(), X_dacTinh.max(), 100)
#    f = theta_final[0, 0] + (theta_final[0, 1] * x_pop)
#    fig, ax = plt.subplots() # (figsize=(12,8))
