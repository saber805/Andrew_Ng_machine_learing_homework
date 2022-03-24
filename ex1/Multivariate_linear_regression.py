import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
ax = plt.axes(projection='3d')
df = pd.read_csv('ex1data2.txt', names=['square', 'bedrooms', 'price'])


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


def get_X(df):#读取特征
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并 axis： 需要合并链接的轴，0是行，1是列
    return data.iloc[:, :-1]


def lr_cost(theta, X, y):

    m = X.shape[0]#m为样本数
    inner = X @ theta - y  # R(m*1)，X @ theta等价于X.dot(theta)
    square_sum = inner.T @ inner
    cost = square_sum / (2 * m)
    return cost


def gradient(theta, X, y):
    m = X.shape[0] #样本个数
    inner = X.T @ (X @ theta - y)  # (m,n).T @ (m, 1) -> (n, 1)，X @ theta等价于X.dot(theta)
    return inner / m


def batch_gradient_decent(theta, X, y, epoch, alpha=0.01):
    cost_data = [lr_cost(theta, X, y)]
    for _ in range(epoch):
        theta = theta - alpha * gradient(theta, X, y)
        cost_data.append(lr_cost(theta, X, y))
    return theta, cost_data


def normalEqn(X, y): #正规方程
    theta = np.linalg.inv(X.T@X)@X.T@y#X.T@X等价于X.T.dot(X)
    return theta


data = normalize_feature(df)  #特征缩放
y = data.values[:, 2]
X = get_X(data)
ax.scatter(X['square'], X['bedrooms'], y, alpha=0.3)
plt.xlabel('square')
plt.ylabel('bedrooms')
ax.set_zlabel(r'$prices$')
epoch = 500
alpha = 0.01
theta = np.zeros(X.shape[1])   #在该问题中X有三个特征(1,square,bedrooms)，所以theta初始为三个零
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch, alpha=alpha)
D = final_theta[0]
A = final_theta[1]
B = final_theta[2]
Z = A*X['square'] + B*X['bedrooms'] + D
ax.plot_trisurf(X['square'], X['bedrooms'], Z,
                       linewidth=0, antialiased=False)

predict_square = float(input('square:'))
predict_square = ((predict_square - df.square.mean())/df.square.std())

predict_bedrooms = float(input('bedrooms'))
predict_bedrooms = ((predict_bedrooms - df.bedrooms.mean())/df.bedrooms.std())

p = A * predict_square + B*predict_bedrooms + D
ax.scatter(predict_square, predict_bedrooms, marker='+', c='red')
p = p * df.price.std() + df.price.mean()
print('I predict the prices is :')
print(p)
plt.show()
