import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# 聚类中心已知，根据数据点距离聚类中心的距离分类
def find_closest_centroids(X, centroids):
    m = X.shape[0]  # X.shape = (300,2)
    k = centroids.shape[0]  # centrids.shape = (3,2)
    idx = np.zeros(m)  # m = 300, 每个数据的标签，默认为0

    for i in range(m):  # m = 300 样本个数
        min_dist = 1000000
        for j in range(k):  # k = 3 聚类中心个数
            dist = np.sum((X[i, :] - centroids[j, :]) ** 2)
            if dist < min_dist:
                min_dist = dist
                idx[i] = j
    return idx  # 返回数据点的标签


# k个聚类中心重新计算均值，返回计算后的坐标
def compute_centroids(X, idx, k):
    m, n = X.shape  # (300,2)
    centroids = np.zeros((k, n))  # （3，2）
    for i in range(k):
        indices = np.where(idx == i)
        centroids[i, :] = (np.sum(X[indices, :], axis=1) / len(indices[0])).ravel()
    return centroids


def run_k_means(X, centroids, max_iters):
    m, n = X.shape  # (300,2)
    k = centroids.shape[0]  # 3
    idx = np.zeros(m)  # 标签初始为0
    for i in range(max_iters):
        idx = find_closest_centroids(X, centroids)  # 给最近的数据做好标签
        centroids = compute_centroids(X, idx, k)  # 重新计算聚类坐标，重复max_iters次
    return idx, centroids


data = loadmat('data/ex7data2.mat')
data2 = pd.DataFrame(data.get('X'), columns=['X1', 'X2'])
# plt.scatter(data2['X1'],data2['X2'],c='b')
# plt.show()

X = data['X']
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])  # 初始化聚类中心
idx, centroids = run_k_means(X, initial_centroids, 10)
cluster1 = X[np.where(idx == 0)[0],:]
cluster2 = X[np.where(idx == 1)[0],:]
cluster3 = X[np.where(idx == 2)[0],:]
plt.scatter(cluster1[:,0], cluster1[:,1], s=30, color='r', label='Cluster 1')
plt.scatter(cluster2[:,0], cluster2[:,1], s=30, color='g', label='Cluster 2')
plt.scatter(cluster3[:,0], cluster3[:,1], s=30, color='b', label='Cluster 3')
plt.legend()
plt.show()


def init_centroids(X, k):
    m, n = X.shape  # (300,2)
    centroids = np.zeros((k, n))  # (3,2),三个聚类中心，每个中心有两个坐标来确定
    idx = np.random.randint(0, m, k)  # 产生k个0~m的数
    for i in range(k):
        centroids[i, :] = X[idx[i], :]  # 将随机选取的三个数据点作为聚类中心
    return centroids