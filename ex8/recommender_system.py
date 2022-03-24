import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat


def serialize(X, theta):  # 展成一维再衔接
    return np.concatenate((X.ravel(), theta.ravel()))


def deserialize(param, n_movie, n_user, n_features):  # 再reshape成向量
    """into ndarray of X(1682, 10), theta(943, 10)"""
    return param[:n_movie * n_features].reshape(n_movie, n_features), \
           param[n_movie * n_features:].reshape(n_user, n_features)


def cost(param, Y, R, n_features):
    """compute cost for every r(i, j)=1
    Args:
        param: serialized X, theta
        Y (movie, user), (1682, 943): (movie, user) rating
        R (movie, user), (1682, 943): (movie, user) has rating
    """
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movie, n_user = Y.shape
    X, theta = deserialize(param, n_movie, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)

    return np.power(inner, 2).sum() / 2


def gradient(param, Y, R, n_features):
    # theta (user, feature), (943, 10): user preference
    # X (movie, feature), (1682, 10): movie features
    n_movies, n_user = Y.shape
    X, theta = deserialize(param, n_movies, n_user, n_features)

    inner = np.multiply(X @ theta.T - Y, R)  # (1682, 943)

    # X_grad (1682, 10)
    X_grad = inner @ theta

    # theta_grad (943, 10)
    theta_grad = inner.T @ X

    # roll them together and return
    return serialize(X_grad, theta_grad)


def regularized_cost(param, Y, R, n_features, l=1):
    reg_term = np.power(param, 2).sum() * (l / 2)

    return cost(param, Y, R, n_features) + reg_term


def regularized_gradient(param, Y, R, n_features, l=1):
    grad = gradient(param, Y, R, n_features)
    reg_term = l * param

    return grad + reg_term


data = loadmat('data/ex8_movies.mat')
# params_data = loadmat('data/ex8_movieParams.mat')
Y = data['Y']  # 评的几颗星，没评为0  1682款电影，943位用户
R = data['R']  # 评分为1，没评为0  shape=(1682,943)

# X = params_data['X'] # 电影参数.shape=(1682,10)
# theta = params_data['Theta']  # 用户参数.shape(943,10)
movie_list = []
with open('data/movie_ids.txt') as f:
    for line in f:
        tokens = line.strip().split(' ')  #去除两端空白再分
        movie_list.append(' '.join(tokens[1:])) # 去掉数字，仅将电影名称存入列表
movie_list = np.array(movie_list)


# 假设我给1682部电影中的某些评了分
ratings = np.zeros(1682)

ratings[0] = 4
ratings[6] = 3
ratings[11] = 5
ratings[53] = 4
ratings[63] = 5
ratings[65] = 3
ratings[68] = 5
ratings[97] = 2
ratings[182] = 4
ratings[225] = 5
ratings[354] = 5

#  插入后ratting将在0的位置（按列插入）Y(评的几颗星),R(评没评分)
# 自己为用户0
Y = np.insert(Y, 0, ratings, axis=1)  # now I become user 0
R = np.insert(R, 0, ratings != 0, axis=1)
movies, users = Y.shape  # (1682,944)
n_features = 50
X = np.random.random(size=(movies, n_features))
theta = np.random.random(size=(users, n_features))
lam = 1
Ymean = np.zeros((movies, 1))
Ynorm = np.zeros((movies, users))
param = serialize(X, theta)
# 均值归一化
for i in range(movies):  #R.shape = (1682,944),movies = 1682
    idx = np.where(R[i,:] == 1)[0]  # 第i部电影，用户0。找到用户0评过分的电影
    Ymean[i] = Y[i,idx].mean()  # 第i行,评过分的求平均
    Ynorm[i,idx] = Y[i,idx] - Ymean[i]  # 评过分的减去他们的均值

res = opt.minimize(fun=regularized_cost,
                   x0=param,
                   args=(Ynorm, R, n_features,lam),
                   method='TNC',
                   jac=regularized_gradient)
X = np.mat(np.reshape(res.x[:movies * n_features], (movies, n_features)))
theta = np.mat(np.reshape(res.x[movies * n_features:], (users, n_features)))
predictions = X * theta.T
my_preds = predictions[:, 0] + Ymean
sorted_preds = np.sort(my_preds, axis=0)[::-1]
idx = np.argsort(my_preds, axis=0)[::-1]

print("Top 10 movie predictions:")
for i in range(10):
    j = int(idx[i])
    print('Predicted rating of {} for movie {}.'.format(str(float(my_preds[j])), movie_list[j]))
