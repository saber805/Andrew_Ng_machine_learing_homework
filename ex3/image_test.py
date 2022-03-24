from PIL import Image
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import matplotlib
import cv2


def produceImage(file_in, width, height, file_out):
    image = Image.open(file_in)
    resized_image = image.resize((width, height), Image.ANTIALIAS)
    resized_image.save(file_out)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def plot_an_image(image):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap=matplotlib.cm.binary)  # 这是一个把矩阵或者数组绘制成图像的函数
    plt.xticks(np.array([]))  # just get rid of ticks
    plt.yticks(np.array([]))


for i in range(10):
    filename = 'image/'+str(i)+'.png'
    im = 255 - np.array(Image.open(filename).convert('L'))
    if im.shape != (20, 20):
        produceImage(filename, 20, 20, filename)
    im = 255 - np.array(Image.open(filename).convert('L').rotate(90))
    im = cv2.flip(im, 0, dst=None)  # 垂直镜像
    im = im.ravel()
    im = im.astype(float) / 255.0
    all_theta = scio.loadmat('theta.mat')
    all_theta = all_theta['all_theta']
    im = np.mat(im)
    im2 = np.insert(im, 0, values=np.ones(1), axis=1)
    answer = sigmoid(im2 * all_theta.T)
    print(" i think this should be " + str(np.argmax(answer) + 1))
    plot_an_image(im)
    plt.show()
