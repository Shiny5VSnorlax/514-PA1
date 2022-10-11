import pandas as pd
import numpy as np
import math
from numpy import linalg as LA
from sklearn.model_selection import train_test_split


def fit(x,w):
    y_pred = x@w
    return y_pred


def linear_regression(y, x, w, func_loss, func_gradient):
    # y_pred = w.t@x
    loss = func_loss(y, x, w)
    gradient = func_gradient(y, x, w)
    return loss, gradient


def mean_square_error_derivative(y, x, w):
    gradient = (2 * x.T @ x @ w - 2 * x.T @ y) / x.shape[0]
    return gradient


def mean_square_error(y, x, w):
    # mse = (1/len(y_pred))*sum((y-y_pred)**2)
    mse = (w.T @ x.T @ x @ w - 2 * w.T @ x.T @ y + y.T @ y) / x.shape[0]
    return mse


def mae_derivative(y, x, w):
    judge = np.sign(x @ w - y)
    gradient = np.sum(np.multiply(judge, x) / x.shape[0], axis=0).reshape(-1, 1)
    return gradient


def mae_loss(y, x, w):
    mae = np.sum(abs(x @ w - y), axis=0) / x.shape[0]
    return mae


def grdescent(func, w0, stepsize, maxiter, tolerance=1e-02):
    eps = 2.2204e-14
    i = 0
    loss, gradient = func(w0)
    w = w0
    while i <= maxiter:
        w = w - stepsize * gradient
        last_loss, last_gradient = loss, gradient
        loss, gradient = func(w)
        if stepsize < eps:
            break
        else:
            if loss < last_loss:
                # print(1)
                stepsize *= 1.01
            else:
                # print(2)
                stepsize *= 0.5
        if LA.norm(gradient) <= tolerance:
            break
        i += 1
    print(i)
    return w
