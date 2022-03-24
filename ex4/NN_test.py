import numpy as np
from scipy.io import loadmat
from PIL import Image


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]  # m = 5000
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)  # input layer
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)  # hidden layer
    z3 = a2 * theta2.T
    h = sigmoid(z3)  # h.shape(5000,10)
    return a1, z2, a2, z3, h


for i in range(10):
    filename = 'image/'+str(i)+'.png'
    im = 255 - np.array(Image.open(filename).convert('L'))  # 为灰度图像，每个像素用8个bit表示，0表示黑，255表示白，

    im = im.ravel()
    im = im.astype(float) / 255.0

    theta1 = loadmat('theta1.mat')
    theta1 = theta1['theta1']

    theta2 = loadmat('theta2.mat')
    theta2 = theta2['theta2']

    im = np.mat(im)
    im2 = np.insert(im, 0, values=np.ones(1), axis=1)
    a1, z2, a2, z3, h = forward_propagate(im, theta1, theta2)
    answer = np.argmax(h) + 1
    if answer == 10:
        answer = 0
    print(" i think this should be " + str(answer))
