from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from k_means import find_closest_centroids, init_centroids, run_k_means
filename = "data/bird_small.png"
im = np.array(Image.open(filename))/255
im2 = np.reshape(im, (im.shape[0]*im.shape[1], im.shape[2]))
initial_centroids = init_centroids(im2, 16)  # 随机选取16个数据点作为聚类中心

idx, centroids = run_k_means(im2, initial_centroids, 10)  # 执行10次k-means算法
idx = find_closest_centroids(im2, centroids)

X_recovered = centroids[idx.astype(int),:]  # X.shape = (16384,3)

X_recovered = np.reshape(X_recovered, (im.shape[0], im.shape[1], im.shape[2]))  # 返回最初的维度

plt.imshow(X_recovered)
plt.show()
