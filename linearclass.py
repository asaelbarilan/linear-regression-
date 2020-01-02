''' version 2,date: 30/12 22:30'''

import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import scipy
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class OrdinaryLinearRegression:
    def __init__(self, Ridge=False, Lambda=1):
        ''' the value of
        ‘ k ’ is chosen small enough, for which the mean squared error of ridge estimator,
        is less than the mean squared error of OLS estimator
        also- MSE with ridge will be smaller than mse without-when 0<lambda<var/weights_max^2
        '''
        self.Ridge = Ridge
        self.Lambda = Lambda
        return

    def fit(self, X, y):
        self.X = X
        self.y = y
        try:
            if not self.Ridge:
                self.weights = np.linalg.pinv(self.X) @ self.y
            # self.weights = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
            else:
                lambda_mat = (self.Lambda * np.eye(self.X.shape[1]))
                lambda_mat[0, 0] = 0
                # self.weights = np.linalg.pinv(self.X+lambda_mat) @ self.y
                self.weights = np.linalg.inv((self.X.T @ self.X) + lambda_mat) @ self.X.T @ self.y

        except Exception:
            print('weight cannot be calculated, use gradient descent regressor')

    def predict(self, X):
        y_pred = X @ self.weights
        return y_pred

    def score(self, y_actual, y_pred):
        N = y_actual.shape[0]
        # MSE = (1 / (N)) * np.sum(np.square(y_actual - y_pred))
        MSEKLEARN = sklearn.metrics.mean_squared_error(y_actual, y_pred)
        return MSEKLEARN


class OLRGradientDescent(OrdinaryLinearRegression):
    def __init__(self, n_features, lr=0.1, n_iteration=1000, treshold=0.05, Ridge=False):
        super().__init__()
        self.lr = lr
        self.n_iteration = n_iteration
        self.treshold = treshold
        self.n_features = n_features
        self.weights_new = np.random.uniform(low=0, high=1, size=self.n_features).reshape(-1, 1)
        self.weights_old = np.random.uniform(low=0, high=1, size=self.n_features).reshape(-1, 1)
        self.weights = 0
        self.grad_loss = []
        self.Ridge = Ridge

    def gradientDescent(self, X, y):
        error = 1
        Loss_old = 0
        y = y.reshape(-1, 1)
        for i in range(self.n_iteration):
            # if error != np.inf and error >= self.treshold:
            self.updateWeights(X, y, self.weights_new)
            Loss_new = 0.5 * (((X @ self.weights_new).reshape(-1, 1) - y).T @ ((X @ self.weights_new).reshape(-1, 1) - y))
            error = abs(Loss_new - Loss_old)
            self.grad_loss.append(Loss_new[0][0])
            if (i % (100)) == 0:
                print('Loss is:', Loss_new[0][0].round(4))
            Loss_old = Loss_new
            # else:
            #     self.weights = self.weights_new
            #     break
        self.weights = self.weights_new
        return

    def updateWeights(self, X, y, w):
        N = X.shape[0]
        self.weight_old = w
        self.weights_new = self.weight_old.reshape(-1, 1) - self.lr * (1 / N) * X.T @ (
                (X @ self.weight_old).reshape(-1, 1) - y)
        return

class OLRCoordinateDescent(OrdinaryLinearRegression):
    def __init__(self, n_features, lr=0.1, n_iteration=1000, treshold=0.05, Ridge=False):
        super().__init__()
        self.lr = lr
        self.n_iteration = n_iteration
        self.treshold = treshold
        self.n_features = n_features
        self.weights_new = np.random.uniform(low=0, high=1, size=self.n_features).reshape(-1, 1)
        self.weights_old = np.random.uniform(low=0, high=1, size=self.n_features).reshape(-1, 1)
        self.weights = np.random.uniform(low=0, high=1, size=self.n_features).reshape(-1, 1)
        self.grad_loss = []
        self.Ridge = Ridge

    def coordinateDescent(self, X, y):
        error = 1
        Loss_old = 0
        y = y.reshape(-1, 1)
        columns = list(range(0,X.shape[1]))
        for j in range(self.n_iteration):
            if error != np.inf and error >= self.treshold:
                for i in range(X.shape[1]):
                    colnotI=columns.copy()
                    colnotI.remove(i)
                    self.weights[i] =  (X[:,i].T @ (y-(X[:,colnotI] @ self.weights[colnotI]).reshape(-1, 1)))/(X[:,i].T @X[:,i])
                Loss_new = 0.5 * (((X @ self.weights).reshape(-1, 1) - y).T @ ((X @ self.weights).reshape(-1, 1) - y))
                error = abs(Loss_new - Loss_old)
                self.grad_loss.append(Loss_new[0][0])
                if (j % (10)) == 0:
                    print('Loss is:', Loss_new[0][0].round(4))
                Loss_old = Loss_new
            else:
                break
        return

    def gradientDescent(self, X, y):
        y = y.reshape(-1, 1)
        for i in range(self.n_iteration):
            self.updateWeights(X, y, self.weights_new)
            Loss= 0.5 * (((X @ self.weights_new).reshape(-1, 1) - y).T @ ((X @ self.weights_new).reshape(-1, 1) - y))
            self.grad_loss.append(Loss[0][0])
            if (i % (100)) == 0:
                print('Loss is:', Loss[0][0].round(4))
        self.weights = self.weights_new
        return

    def updateWeights(self, X, y, w):
        N = X.shape[0]
        self.weight_old = w.reshape(-1, 1)
        w0= w.reshape(-1, 1)
        self.weights_new = w0 - self.lr * (1 / N) * X.T @ ((X @ self.weight_old).reshape(-1, 1) - y)
        return


#######################Lasso##########################


if __name__ == "__main__":
    '''loading data'''
    X, y = load_boston(return_X_y=True)
    y = y.reshape(-1, 1)

    '''splitting data train and test'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
                                                        random_state=10)
    '''normalizing and scaling '''
    ssx = StandardScaler().fit(X_train)
    X_train_std = ssx.transform(X_train)
    X_test_std = ssx.transform(X_test)
    ssy = StandardScaler().fit(y_train)
    y_train_std = ssy.transform(y_train)
    y_test_std = ssy.transform(y_test)

    '''preproccessing-adding column for bias term '''
    ones = np.ones(X_train_std.shape[0]).reshape(-1, 1)
    X_train_std = np.concatenate((ones, X_train_std), axis=1)
    ones = np.ones(X_test_std.shape[0]).reshape(-1, 1)
    X_test_std = np.concatenate((ones, X_test_std), axis=1)

    # no ridge
    # train
    Lin = OrdinaryLinearRegression(Ridge=False)
    Lin.fit(X_train_std, y_train_std)
    y_pred_train_std = Lin.predict(X_train_std)
    y_pred_train = ssy.inverse_transform(y_pred_train_std)
    Base_MSE_train = Lin.score(y_train, y_pred_train)
    # test
    y_pred_test_std = Lin.predict(X_test_std)
    y_pred_test = ssy.inverse_transform(y_pred_test_std)
    Base_MSE_test = Lin.score(y_test, y_pred_test)
    # with ridge
    Lambda_list = np.arange(0, 2, 0.001)
    MSE_list = []
    for lambdaval in Lambda_list:
        Lin = OrdinaryLinearRegression(Ridge=True, Lambda=lambdaval)
        Lin.fit(X_train_std, y_train_std)
        # y_pred_train_std = Lin.predict(X_train_std)
        # y_pred_train = ssy.inverse_transform(y_pred_train_std)
        # Base_MSE_train_ridge = Lin.score(y_train, y_pred_train)
        # test
        y_pred_test_std = Lin.predict(X_test_std)
        y_pred_test = ssy.inverse_transform(y_pred_test_std)
        Base_MSE_test_ridge = Lin.score(y_test, y_pred_test)
        MSE_list.append(Base_MSE_test_ridge)
    plt.plot(Lambda_list, MSE_list, '.')
    plt.xlabel('Lambda_list')
    plt.ylabel('MSE ')
    plt.title('Ridge ')
    plt.show()

    plt.plot(y_train, y_pred_train, '.')
    plt.xlabel('y train')
    plt.ylabel('y prediction ')
    plt.title('OLS train vs prediction ')
    plt.show()

    '''why there different MSE's for train and test:
        e_train_mean=y_train_mean-x_train_mean*beta_train
        and for  test
        e_test_mean=y_test_mean-x_test_mean*beta_train
        so we get linear combination of predictions mean and data mean
        that are different for train and test  and also betas are from train so
        they are noninclusive for the test

       Err(X0)=σ2ϵ+[Ef^(X0)−f(X0)]2+E[f^(X0)−Ef^(X0)]2
       we have=irreducalbe error+bias^2+variance
        '''
    mse_train_vector = []
    mse_test_vector = []
    for i in range((20)):
        Lin = OrdinaryLinearRegression()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)  #
        Lin.fit(X_train, y_train)
        y_pred_train = Lin.predict(X_train)
        y_pred_test = Lin.predict(X_test)
        mse_train_vector.append(Lin.score(y_train, y_pred_train))
        mse_test_vector.append(Lin.score(y_test, y_pred_test))

    _, p = scipy.stats.ttest_rel(mse_train_vector, mse_test_vector)
    if p < 0.1:
        print('p is:{} so train MSE mean is much smaller than test MSE'.format(p))

    # Gradient descent
    GDLin = OLRGradientDescent(n_features=X_train_std.shape[1], lr=0.05)
    GDLin.gradientDescent(X_train_std, y_train_std)
    y_pred_test_gd = GDLin.predict(X_test_std)
    y_pred_test_gd_trans = ssy.inverse_transform(y_pred_test_gd)
    MSE = GDLin.score(y_test.reshape(-1, 1), y_pred_test_gd_trans)

    print(MSE, 'GD test MSE')

    n_iteration = 1000
    plt.plot(GDLin.grad_loss, list(range(n_iteration)), '.')
    plt.xlabel('iterations')
    plt.ylabel('loss ')
    plt.title('GD loss')
    plt.show()

    # Coordinate descent
    CDLin = OLRCoordinateDescent(n_features=X_train_std.shape[1], lr=0.05)
    CDLin.coordinateDescent(X_train_std, y_train_std)
    y_pred_test_gd = CDLin.predict(X_test_std)
    y_pred_test_gd_trans = ssy.inverse_transform(y_pred_test_gd)
    MSE = CDLin.score(y_test.reshape(-1, 1), y_pred_test_gd_trans)

    print(MSE, 'CD test MSE')

    plt.plot(CDLin.grad_loss, list(range(len(CDLin.grad_loss))), '.')
    plt.xlabel('iterations')
    plt.ylabel('loss ')
    plt.title('CD loss')
    plt.show()

#########3###############

# Lasso
