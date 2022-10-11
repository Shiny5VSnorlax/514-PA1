import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from linear_regression import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
test = 1
raw = 1
##### processed data
if (not raw):
    y = pd.read_csv("../data/processed/y.csv")
    x_raw = pd.read_csv("../data/raw/x_raw.csv")
    X = pd.read_csv("../data/processed/x.csv")

    train_x_raw, test_x_raw, train_y, test_y = train_test_split(x_raw, y, test_size=13 / 103, random_state=42)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=13 / 103, random_state=42)
    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    test_y = test_y.values

#### raw data
if(raw):
    y = pd.read_csv("../data/processed/y.csv")
    X = pd.read_csv("../data/raw/x_raw.csv")
    print(y)


    #train_x_raw, test_x_raw, train_y, test_y = train_test_split(x_raw, y, test_size=13 / 103, random_state=42)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=13 / 103, random_state=42)
    train_x = train_x.values
    train_y = train_y.values
    test_x = test_x.values
    test_y = test_y.values
#### uni
if (not test):
    m = train_x.shape[0]

    for i in range(train_x.shape[1]):

        train_x_uni = train_x[:, i].reshape((m, 1))
        # m = train_x.shape[0]
        x0 = np.ones((m, 1))
        train_x_uni = np.hstack((x0, train_x_uni))
        w0 = np.matrix("0;0")
        # w0 = np.array([1,1])
        f = lambda w: linear_regression(train_y, train_x_uni, w, mean_square_error, mean_square_error_derivative)
        f_mae = lambda w: linear_regression(train_y, train_x_uni, w, mae_loss, mae_derivative)
        #0.01
        w = grdescent(f, w0, 0.00001, 10000)
        w_mae = grdescent(f_mae,w0,0.00001,10000)
        r2_mse = r2_score(train_y,fit(train_x_uni,w))
        r2_mae = r2_score(train_y,fit(train_x_uni,w_mae))
        print("The current feature is: ",X.columns[i])
        print("For uni model with processed data, the train mse is ", mean_square_error(train_y,train_x_uni,w) ,"\nthe cofficient are:\n ",w,"\n")
        print("For uni model with processed data, the train mae is ", mae_loss(train_y,train_x_uni,w_mae) ,"\nthe cofficient are:\n ",w_mae,"\n")
        print("For uni model with processed data, the mse r2 is ", r2_mse,
              "\nthe mae r2 is:\n ", r2_mae, "\n")


    #### multi
    m = train_x.shape[0]
    train_x_multi = train_x.reshape((m, 8))
    # m = train_x.shape[0]
    x0 = np.ones((m, 1))
    train_x_multi = np.hstack((x0, train_x_multi))
    w0 = np.matrix("1;1;1;1;1;1;0;0;1")
    f = lambda w: linear_regression(train_y, train_x_multi, w, mean_square_error, mean_square_error_derivative)
    f_mae = lambda w: linear_regression(train_y, train_x_multi, w, mae_loss, mae_derivative)
    w_multi = grdescent(f, w0, 0.00001, 10000)
    w_multi_mae = grdescent(f_mae,w0,0.00001,10000)
    r2_multi_mse = r2_score(train_y,fit(train_x_multi,w_multi))
    r2_multi_mae = r2_score(train_y,fit(train_x_multi,w_multi_mae))
    print("For multi model with processed data, the train mse is ",mean_square_error(train_y,train_x_multi,w_multi) ,"\nthe cofficient are:\n ",w_multi,"\n")
    print("For multi model with processed data, the train mae is ",mae_loss(train_y,train_x_multi,w_multi_mae) ,"\nthe cofficient are:\n ",w_multi_mae,"\n")
    print("For multi model with processed data, the mse r2 is ", r2_multi_mse,
              "\nthe mae r2 is:\n ", r2_multi_mae, "\n")

    headers = []

##############
#### Test data
#### uni
if (test):
    m = test_x.shape[0]

    for i in range(test_x.shape[1]):

        test_x_uni = test_x[:, i].reshape((m, 1))
        # m = train_x.shape[0]
        x0 = np.ones((m, 1))
        test_x_uni = np.hstack((x0, test_x_uni))
        w0 = np.matrix("1;1")
        # w0 = np.array([1,1])
        f = lambda w: linear_regression(test_y, test_x_uni, w, mean_square_error, mean_square_error_derivative)
        f_mae = lambda w: linear_regression(test_y, test_x_uni, w, mae_loss, mae_derivative)
        w = grdescent(f, w0, 0.00001, 10000)
        w_mae = grdescent(f_mae,w0,0.00001,10000)
        r2_mse = r2_score(test_y, fit(test_x_uni, w))
        r2_mae = r2_score(test_y, fit(test_x_uni, w_mae))
        print("The current feature is: ",X.columns[i])
        print("For uni model with processed data, the test mse is ", mean_square_error(test_y, test_x_uni, w), "\nthe cofficient are:\n ", w, "\n")
        print("For uni model with processed data, the test mae is ", mae_loss(test_y, test_x_uni, w_mae), "\nthe cofficient are:\n ", w_mae, "\n")
        print("For uni model with processed data, the mse r2 is ", r2_mse,
              "\nthe mae r2 is:\n ", r2_mae, "\n")


    #### multi
    m = test_x.shape[0]
    test_x_multi = test_x.reshape((m, 8))
    # m = train_x.shape[0]
    x0 = np.ones((m, 1))
    test_x_multi = np.hstack((x0, test_x_multi))
    w0 = np.matrix("1;1;1;1;1;1;1;1;1")
    f = lambda w: linear_regression(test_y, test_x_multi, w, mean_square_error, mean_square_error_derivative)
    f_mae = lambda w: linear_regression(test_y, test_x_multi, w, mae_loss, mae_derivative)
    w_multi = grdescent(f, w0, 0.00001, 10000)
    w_multi_mae = grdescent(f_mae,w0,0.00001,10000)
    r2_multi_mse = r2_score(test_y, fit(test_x_multi, w_multi))
    r2_multi_mae = r2_score(test_y, fit(test_x_multi, w_multi_mae))
    print("For multi model with processed data, the test mse is ", mean_square_error(test_y, test_x_multi, w_multi), "\nthe cofficient are:\n ", w_multi, "\n")
    print("For multi model with processed data, the test mae is ", mae_loss(test_y, test_x_multi, w_multi_mae), "\nthe cofficient are:\n ", w_multi_mae, "\n")
    print("For multi model with processed data, the mse r2 is ", r2_multi_mse,
              "\nthe mae r2 is:\n ", r2_multi_mae, "\n")

##### raw data