import pandas as pd
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report#这个包是评价报告


def get_X(df):#读取特征
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并
    return data.iloc[:, :-1].values


def get_y(df):#读取标签
    return np.array(df.iloc[:, -1])#df.iloc[:, -1]是指df的最后一列


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())#特征缩放


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    return np.mean(-y * np.log(sigmoid(X @ theta)) - (1 - y) * np.log(1 - sigmoid(X @ theta)))


def gradient(theta, X, y):
    return (1 / len(X)) * X.T @ (sigmoid(X @ theta) - y)


def predict(x, theta):
    prob = sigmoid(x @ theta)
    return (prob >= 0.5).astype(int)#.astype方法前的数据类型为numpy.ndarray


theta=np.zeros(3)
data = pd.read_csv('ex2data1.txt', names=['exam 1', 'exam 2', 'admitted'])
X = get_X(data)
y = get_y(data)
res = opt.minimize(fun=cost, x0=theta, args=(X, y), method='Newton-CG', jac=gradient)
'''func：优化的目标函数
x0：初值
fprime：提供优化函数func的梯度函数，不然优化函数func必须返回函数值和梯度，或者设置approx_grad=True
args：元组，是传递给优化函数的参数'''
final_theta = res.x#res.x is final theta
y_pred = predict(X, final_theta)

#二维图
positive = data[data['admitted'].isin([1])]
negative = data[data['admitted'].isin([0])]
plt.scatter(positive['exam 1'], positive['exam 2'], s=50, c='b', marker='o', label='Admitted')
plt.scatter(negative['exam 1'], negative['exam 2'], s=50, c='r', marker='x', label='Not Admitted')
plt.legend(loc=0)#增加图例,0,1,2,3,4分别表示不同的位置
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')
coef = -(res.x / res.x[2])  # find the equation
x = np.arange(130, step=0.1)
y2 = coef[0] + coef[1]*x#y = theta@X = sigmod(theta[0]*X[0]+theta[1]*X[1]+theta[2]*X[2])
plt.plot(x, y2, 'grey')
plt.show()


#三维图，注释掉了，看三维效果把下面的注释去掉，49-60加注释
"""x = plt.axes(projection='3d')
ax.scatter(X[:,1], X[:,2], y, alpha=0.3)
D = final_theta[0]
A = final_theta[1]
B = final_theta[2]
Z = A*X[:,1] + B*X[:,2] + D
ax.plot_trisurf(X[:,1], X[:,2], Z,
                       linewidth=0, antialiased=False,color='r')
ax.set_zlim(-2,2);"""
print(classification_report(y, y_pred))


