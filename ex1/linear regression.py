import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('ex1data1.txt', names=['population', 'profit'])
data = df


#def normalize_feature(df):
    #return df.apply(lambda column: (column - column.mean()) / column.std())#特征缩放


def get_X(df):#读取特征
    ones = pd.DataFrame({'ones': np.ones(len(df))})#ones是m行1列的dataframe
    data = pd.concat([ones, df], axis=1)  # 合并数据，根据列合并 axis： 需要合并链接的轴，0是行，1是列
    return data.iloc[:, :-1]#返回所有的行，清除最后一列。逗号前为行，后为列


def linear_cost(theta , X , y):
    m = X.shape[0]  #样本数
    inner = X @ theta - y   #与目标的差值即h(theta),inner算出来为一行
    square_sum = inner.T @ inner    #h(theta)的平方
    cost = square_sum/(2*m)
    return cost


def gradient(theta, X, y):
    m = X.shape[0]
    inner = X.T @ (X@theta - y) #X仅有仅有一个特征，恒为1的不算，即该语句算的是更新theta1时，损失函数对theta1的求导
    return inner/m


def batch_gradient_decent(theta, X, y, epoch, alpha=0.02):
    cost_data = [linear_cost(theta, X, y)]
    for _ in range(epoch):   #_仅是一个循环标志，在循环中不会用到
        theta = theta - alpha * gradient(theta, X, y)
        cost_data.append(linear_cost(theta, X, y))
    return theta, cost_data


X = get_X(df)
y = df.values[:, 1]
theta = np.zeros(df.shape[1])
epoch = 6000
final_theta, cost_data = batch_gradient_decent(theta, X, y, epoch)
b = final_theta[0]
k = final_theta[1]
plt.scatter(data.population, data.profit, label="Training data")
plt.plot(data.population, data.population*k + b, label="Prediction")
plt.xlabel('population')
plt.ylabel('profit')
plt.legend(loc=2)


# forecast = float(input('population'))
# predict_profit = forecast*k+b
# print(predict_profit)
# plt.scatter(forecast, predict_profit, marker='+', c='red')
# plt.show()
