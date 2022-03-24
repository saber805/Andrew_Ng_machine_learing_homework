from skimage import io
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans#导入kmeans库
# cast to float, you need to do this otherwise the color would be weird after clustring
pic = io.imread('data/bird_small.png') / 255.

data = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])

model = KMeans(n_clusters=16, n_init=100, n_jobs=-1)  # n_clusters聚类中心个数
#  n_init用不同的质心初始化值运行算法的次数，最终解是在inertia意义下选出的最优结果。
# n_jobs：整形数。　指定计算所用的进程数。内部原理是同时进行n_init指定次数的计算。
# （１）若值为 -1，则用所有的CPU进行运算。若值为1，则不进行并行运算，这样的话方便调试。
# （２）若值小于-1，则用到的CPU数为(n_cpus + 1 + n_jobs)。因此如果 n_jobs值为-2，则用到的CPU数为总CPU数减1。
model.fit(data)

centroids = model.cluster_centers_
C = model.predict(data)
compressed_pic = centroids[C].reshape((128,128,3))
fig, ax = plt.subplots(1, 2)
ax[0].imshow(pic)
ax[1].imshow(compressed_pic)
plt.show()
