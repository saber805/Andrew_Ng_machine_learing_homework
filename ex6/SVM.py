import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn import svm

raw_data = loadmat('data/ex6data1.mat')
data = pd.DataFrame(raw_data['X'], columns=['X1', 'X2'])
data['y'] = raw_data['y']
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([0])]
'''
plt.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
plt.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
plt.legend()

plt.show()
'''
svc = svm.LinearSVC(C=1, loss='hinge', max_iter=1000)
svc.fit(data[['X1', 'X2']], data['y'])
svc.score(data[['X1', 'X2']], data['y'])
data['SVM 1 Confidence'] = svc.decision_function(data[['X1', 'X2']])
plt.scatter(data['X1'], data['X2'], s=50, c=data['SVM 1 Confidence'], cmap='seismic')
plt.title('SVM (C=1) Decision Confidence')
#plt.savefig("SVM2.png")
plt.show()


svc2 = svm.LinearSVC(C=100, loss='hinge', max_iter=1000)
svc2.fit(data[['X1', 'X2']], data['y'])
svc2.score(data[['X1', 'X2']], data['y'])
data['SVM 2 Confidence'] = svc2.decision_function(data[['X1', 'X2']])
plt.scatter(data['X1'], data['X2'], s=50, c=data['SVM 2 Confidence'], cmap='seismic')
plt.title('SVM (C=100) Decision Confidence')
plt.savefig("SVM3.png")
plt.show()
