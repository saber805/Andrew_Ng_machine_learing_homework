import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y, learningRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (learningRate / (2 * len(X))) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(first - second) / len(X) + reg


def gradientReg(theta, X, y, learningRate):
    theta = np.mat(theta)
    X = np.mat(X)
    y = np.mat(y)
    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)
    error = sigmoid(X * theta.T) - y
    for i in range(parameters):
        term = np.multiply(error, X[:, i])
        if (i == 0):
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((learningRate / len(X)) * theta[:, i])
    return grad


def predict(theta, X):
    probability = sigmoid(X * theta.T)
    return [1 if x >= 0.5 else 0 for x in probability]


data2 = pd.read_csv("ex2data2.txt", header=None, names=['Test 1', 'Test 2', 'Accepted'])
df = data2[:]
# 原始数据
positive = data2[data2['Accepted'].isin([1])]
negative = data2[data2['Accepted'].isin([0])]
plt.scatter(positive['Test 1'], positive['Test 2'], s=50, c='b', marker='o', label='Accepted')
plt.scatter(negative['Test 1'], negative['Test 2'], s=50, c='r', marker='x', label='Rejected')
plt.legend()
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')
plt.show()


degree = 5
x1 = data2['Test 1']
x2 = data2['Test 2']
data2.insert(3, 'Ones', 1)  # 插入在第三列之前
for i in range(1, degree):  # i:1~4
    for j in range(0, i):  # i=4时，j为0到3
        data2['F' + str(i) + str(j)] = np.power(x1, i-j) * np.power(x2, j)
'''将F10，F20,F21这些列加在data2的最后，例如：data["ww"]=1    会在data2的列之后在格外加一列名字为ww'''
data2.drop('Test 1', axis=1, inplace=True)  # 删除方法，将index为Test 1，axis=1说明删除的是列
data2.drop('Test 2', axis=1, inplace=True)  # 如果手动设定为True（默认为False），那么原数组直接就被替换
# 而采用inplace=False之后，原数组名对应的内存值并不改变，需要将新的结果赋给一个新的数组或者覆盖原数组的内存位置
cols = data2.shape[1]
X2 = data2.iloc[:, 1:cols].values  # 删除accepted列
y2 = data2.iloc[:, 0:1].values  # 仅保留accepted列
theta2 = np.zeros(11)
learningRate = 1
final_theta = opt.fmin_tnc(func=cost, x0=theta2, fprime=gradientReg, args=(X2, y2, learningRate))
'''或用
res = opt.minimize(fun=cost,
                       x0=theta2,
                       args=(X2, y2, learningRate),
                       method='TNC',
                       jac=gradientReg)
final_theta = res.x可得到相同的结果

func：优化的目标函数
x0：初值
fprime：提供优化函数func的梯度函数，不然优化函数func必须返回函数值和梯度，或者设置approx_grad=True
args：元组，是传递给优化函数的参数
'''
theta_min = np.mat(final_theta[0])
predictions = predict(theta_min, X2)
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0 for (a, b) in zip(predictions, y2)]
accuracy = (sum(map(int, correct)) % len(correct))
print('accuracy = {0}%'.format(accuracy))
'''画出决策边界，由于𝑋×𝜃是个11维的图，我们不能直观的表示出来
但可以找到所有 𝑋×𝜃近似等于0的值以此来画出决策边界'''

