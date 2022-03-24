import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm
spam_train = loadmat('data/spamTrain.mat')
spam_test = loadmat('data/spamTest.mat')
X = spam_train['X']
Xtest = spam_test['Xtest']
y = spam_train['y']
ytest = spam_test['ytest']
svc = svm.SVC()
svc.fit(X, y)
print('Test accuracy = {0}%'.format(np.round(svc.score(Xtest, ytest) * 100)),2)  # np.round()四舍五入,此处保留两位小数

