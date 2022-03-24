import scipy.io as scio
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd


def load_data():
    d = scio.loadmat('ex5data1.mat')
    return map(np.ravel, [d['X'], d['y'], d['Xval'], d['yval'], d['Xtest'], d['ytest']])
X, y, Xval, yval, Xtest, ytest = load_data()


def cost(theta, X, y):
    m = X.shape[0]  # m=12
    inner = X @ theta - y  # R(m*1),X(12,2),theta(2,1)
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost


def gradient(theta, X, y):  # 梯度，cost的导数
    m = X.shape[0]  # 12
    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)
    return inner / m


def regularized_gradient(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = theta.copy()  # 直接赋值会让两个变量指向一个值
    regularized_term[0] = 0  # don't regularize intercept theta
    regularized_term = (l / m) * regularized_term
    return gradient(theta, X, y) + regularized_term


def regularized_cost(theta, X, y, l=1):
    m = X.shape[0]
    regularized_term = (l / (2 * m)) * np.power(theta[1:], 2).sum()
    return cost(theta, X, y) + regularized_term


def linear_regression_np(X, y, l=1):
    theta = np.ones(X.shape[1])
    res = opt.minimize(fun=regularized_cost,
                       x0=theta,
                       args=(X, y, l),
                       method='TNC',
                       jac=regularized_gradient,
                       options={'disp': True})
    return res


def poly_features(x, power, as_ndarray=False):
    data = {'f{}'.format(i): np.power(x, i) for i in range(1, power + 1)}  # 将x拓展为x,x^2,x^3
    df = pd.DataFrame(data)
    return df.values if as_ndarray else df


def normalize_feature(df):  # 特征缩放
    return df.apply(lambda column: (column - column.mean()) / column.std())  # lambda函数：前为输入，：后为输出


def prepare_poly_data(*args, power):
    def prepare(x):
        # expand feature
        df = poly_features(x, power=power)  # 将特征向高维拓展
        # normalization
        ndarr = normalize_feature(df).values
        # add intercept term
        return np.insert(ndarr, 0, np.ones(ndarr.shape[0]), axis=1)
    return [prepare(x) for x in args]


def plot_learning_curve(X, y, Xval, yval, l=0):
    training_cost, cv_cost = [], []
    m = X.shape[0]

    for i in range(1, m + 1):
        # regularization applies here for fitting parameters
        res = linear_regression_np(X[:i, :], y[:i], l=l)

        # remember, when you compute the cost here, you are computing
        # non-regularized cost. Regularization is used to fit parameters only
        tc = cost(res.x, X[:i, :], y[:i])
        cv = cost(res.x, Xval, yval)

        training_cost.append(tc)
        cv_cost.append(cv)

    plt.plot(np.arange(1, m + 1), training_cost, label='training cost')
    plt.plot(np.arange(1, m + 1), cv_cost, label='cv cost')
    plt.legend(loc=1)

'''
theta = np.ones(X.shape[1])
final_theta = linear_regression_np(X, y, l=0).get('x')
b = final_theta[0] # intercept
m = final_theta[1] # slope

plt.scatter(X[:,1], y, label="Training data")
plt.plot(X[:, 1], X[:, 1]*m + b, label="Prediction")
plt.legend(loc=2)

plt.savefig('linear.jpg')
plt.show()

training_cost, cv_cost = [], []
m = X.shape[0]  # m =12
for i in range(1, m + 1):
    res = linear_regression_np(X[:i, :], y[:i], l=0)
    tc = regularized_cost(res.x, X[:i, :], y[:i], l=0)
    cv = regularized_cost(res.x, Xval, yval, l=0)
    training_cost.append(tc)
    cv_cost.append(cv)

plt.plot(np.arange(1, m+1), training_cost, label='training cost')
plt.plot(np.arange(1, m+1), cv_cost, label='cv cost')
plt.legend(loc=1)
plt.savefig('linear_cost&cv_cost.jpg')
plt.show()



plot_learning_curve(X_poly, y, Xval_poly, yval, l=0)
plt.savefig('overfit.jpg')
plt.show()
'''
X_poly, Xval_poly, Xtest_poly = prepare_poly_data(X, Xval, Xtest, power=8)  # 所有数据集拓展
l_candidate = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
training_cost, cv_cost = [], []
for l in l_candidate:
    res = linear_regression_np(X_poly, y, l)

    tc = cost(res.x, X_poly, y)
    cv = cost(res.x, Xval_poly, yval)

    training_cost.append(tc)
    cv_cost.append(cv)
plt.plot(l_candidate, training_cost, label='training')
plt.plot(l_candidate, cv_cost, label='cross validation')
plt.legend(loc=2)

plt.xlabel('lambda')

plt.ylabel('cost')
plt.savefig('finally.png')
plt.show()