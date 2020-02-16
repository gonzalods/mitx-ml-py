import numpy as np


def perceptron(X, labels, T=20, converge=True, theta=None, theta_0=None, Obj=False):
    if theta is None:
        theta = np.array([0.0, 0.0])
    if theta_0 is None:
        theta_0 = 0.0
    errores = [0,0,0,0,0]
    objetivo = [1,0,2,1,0]
    #parametros = []
    for t in range(T):
        err = 0
        for indx, (x, y) in enumerate(list(zip(X, labels))):
            prediccion = np.matmul(theta, x) + theta_0
            signo = y*prediccion
            if signo <= 0:
                err += 1
                theta = theta + y*x 
                theta_0 = theta_0 + y
                errores[indx] += 1
        if converge and err == 0:
            break
        if Obj and errores == objetivo:
            break
          
    return theta, theta_0

theta = np.array([0.0, 0.0])
X = np.array([
    [-4.0, 2.0],
    [-2.0, 1.0],
    [-1.0, -1.0],
    [2.0, 2.0],
    [1.0, -2.0],
])
labels= np.array([
    [1],
    [1],
    [-1],
    [-1],
    [-1]
])

theta, theta_0 = perceptron(X, labels)
print('theta:',theta)
print('theta_0:',theta_0)