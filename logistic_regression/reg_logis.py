
# coding: utf-8

# In[2]:

import numpy as np
import math
import time
import matplotlib.pyplot as plt
from sklearn import linear_model
import scipy.stats as st


# ## Mean Squared Error
# $MSE(\hat{w})=\frac{1}{N}(y-\hat{\mathbf{w}}^T\mathbf{x})^T(y-\hat{\mathbf{w}}^T\mathbf{x})$

# In[3]:

def compute_mse_vectorized(w, X, Y):
    '''This function returns de MSE for a given dataset and coefficients'''
    res = Y - np.dot(X, w)
    totalError = np.dot(res.T, res)
    return totalError / float(len(Y))


# ## Regressão Logística Vetorizada

# In[25]:

def step_gradient_vectorized(w_current, X, Y, alpha):
    '''This function calculates the step gradient using alpha value as stepsize.'''
    w = w_current
    # valores previstos com o vetor de coeficientes atual
    Y_pred = np.dot(X, w)
    res = np.subtract(Y, Y_pred)  # resíduos entre Y observados e Y previstos
    gradient_rss = -2 * np.dot(X.T, res)  # vetor de derivadas parciais
    new_w = np.add(w, alpha * (gradient_rss))
    return [new_w, gradient_rss]


# In[ ]:

def gradient_ascent_runner_vectorized(starting_w, X, Y, learning_rate, epsilon):
    '''This function returns the coefficients' vector'''
    w = starting_w
    grad = np.array([0, 0, 0, 0])
    i = 0
    while(np.linalg.norm(grad) <= epsilon):
        w, grad = step_gradient_vectorized(w, X, Y, learning_rate)
        # if i % 10 == 0:
        print("MSE na iteração {0} é de {1}".format(
            i, compute_mse_vectorized(w, X, Y)))
        print("grad norm: {0}".format(np.linalg.norm(grad)))
        i += 1
    return w


# In[6]:

points = np.genfromtxt("../data/iris.csv", delimiter=",", dtype="str")
points = points[1:]


# In[7]:

def predict(Y):
    '''It sets the classes's values to 0 or 1 integers'''
    levels = np.unique(Y)
    resp = np.where(Y == levels[0], 0, 1)
    return resp


# In[8]:

X = points[:, [0, 1, 2, 3]].astype("float")
Y = predict(points[:, [4]])


# In[59]:

num_coeficients = X.shape[1]
init_w = np.zeros((num_coeficients, 1))

learning_rate = 0.0000001
epsilon = 50000

# In[58]:

start_grad_asc_runner_time = time.time()
w = gradient_ascent_runner_vectorized(init_w, X, Y, learning_rate, epsilon)
end_grad_asc_runner_time = time.time()
print('time: {0}'.format(
    end_grad_asc_runner_time - start_grad_asc_runner_time))


# In[56]:

# The coefficients
print('Coefficients: \n', w)


# start_sklearn_reg = time.time()
# # Create linear regression object
# regr = linear_model.LogisticRegression()
# # Train the model using the training sets
# regr.fit(X, Y)
# end_sklearn_reg = time.time()
# print('time: {0}'.format(end_sklearn_reg - start_sklearn_reg))
# # The coefficients
# print('Coefficients: \n', regr.coef_.T)
