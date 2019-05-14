import numpy as np

def h_X(X, theta):
    return np.dot(X, theta.T)

def computeCost(X, y, theta, m):
    err = np.power((h_X(X, theta) - y),2)
    J = (1.0/(2*m)) * np.sum(err)

    return J

def gradientDescent(X, y, theta, m, alpha = 0.01, loop = 10):
    '''
    X: data input không có cột cuối (Price)
    y: cột cuối (Price)
    m: số lượng data input
    alpha: learning rate
    loop: số lần lặp của thuật toán
    '''
    theta_temp = np.zeros(theta.shape)
    var_Theta = theta.shape[1]
    J_temp = np.zeros(loop)
    for i in range(loop):
        err = h_X(X,theta) - y
        for j in range(var_Theta):
            x_ij = np.reshape(X[:,j],(len(X),1))
            term = np.multiply(err,x_ij)
            theta_temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        theta = theta_temp
        J_temp[i] = computeCost(X,y,theta,m)
    
    return theta, J_temp
