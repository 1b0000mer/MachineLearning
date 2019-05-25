import numpy as np

def h_X(X, theta):
    '''
    hypothesis
    '''
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
        loss = h_X(X,theta) - y
        for j in range(var_Theta):
            x_ij = np.reshape(X[:,j],(len(X),1))
            term = np.multiply(loss,x_ij)
            theta_temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
        theta = theta_temp
        J_temp[i] = computeCost(X,y,theta,m)
    #for i in range(0, loop):
    #    hypothesis = h_X(X, theta)
    #    loss = hypothesis - y
    #    # avg cost per example (the 2 in 2*m doesn't really matter here.
    #    # But to be consistent with the gradient, I include it)
    #    cost = np.sum(loss ** 2) / (2 * m)
    #    #print("Iteration %d | Cost: %f" % (i, cost))
    #    # avg gradient per example
    #    gradient = np.dot(X.T, loss) / m
    #    # update
    #    theta = theta - alpha * gradient
    
    return theta, J_temp
