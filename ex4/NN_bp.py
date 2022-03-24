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


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]  # m = 5000
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)  # input layer
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)  # hidden layer
    z3 = a2 * theta2.T
    h = sigmoid(z3)  # h.shape(5000,10)
    return a1, z2, a2, z3, h


for i in range(10):
    filename = 'image/'+str(i)+'.png'  #识别这里的图片
    im = 255 - np.array(Image.open(filename).convert('L'))
    if im.shape != (20, 20):
        produceImage(filename, 20, 20, filename)
    im = 255 - np.array(Image.open(filename).convert('L').rotate(90))
    im = cv2.flip(im, 0, dst=None)  # 垂直镜像
    im = im.ravel()
    im = im.astype(float) / 255.0

    theta1 = scio.loadmat('theta1.mat')
    theta1 = theta1['theta1']

    theta2 = scio.loadmat('theta2.mat')
    theta2 = theta2['theta2']
    im = np.mat(im)
    im2 = np.insert(im, 0, values=np.ones(1), axis=1)
    a1, z2, a2, z3, h = forward_propagate(im, theta1, theta2)
    print(" i think this should be " + str(np.argmax(h) + 1))
