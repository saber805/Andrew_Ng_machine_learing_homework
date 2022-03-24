import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.io import loadmat
import math
data = loadmat('data/ex8data1.mat')  # Xval,yval,X
X = data['X']
Xval = data['Xval']
yval = data['yval']
# plt.scatter(X[:,0], X[:,1])
# plt.show()


def estimate_gaussian(X):
    mu = X.mean(axis=0)  # X.mean()所有数取均值，X.mean(axis=0)取所有列的均值
    sigma = X.var(axis=0)  # 方差
    return mu, sigma


def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0

    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon
        #  np.logical_and逻辑与
        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1


mu, sigma = estimate_gaussian(X)  # X每一个特征的均值（mu）和方差(sigma)


p = np.zeros((X.shape[0], X.shape[1]))  # X.shape = (307,2)
p0 = stats.norm(mu[0], sigma[0])
p1 = stats.norm(mu[1], sigma[1])
p[:,0] = p0.pdf(X[:,0])
p[:,1] = p1.pdf(X[:,1])


pval = np.zeros((Xval.shape[0], Xval.shape[1]))  # Xval.shape = (307,2)
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])  # pval的第0列为验证集的第一个特征带入根据X第一个特征建立的高斯函数值
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])
epsilon, f1 = select_threshold(pval, yval)  # 用pval得到最佳阈值，再用阈值epsilon寻找p中的异常值
outliers = np.where(p < epsilon)
plt.scatter(X[:,0], X[:,1])
plt.scatter(X[outliers[0],0], X[outliers[0],1], s=50, color='r', marker='o')


def gussian(x,mu=mu[0], sigma=sigma[0]):
    s = 1/(math.sqrt(2*math.pi)*sigma)
    ss = np.exp(-1*(np.power((x-mu),2))/(2*np.power(sigma,2)))
    return s*ss

