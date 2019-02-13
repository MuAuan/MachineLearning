import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import datasets, metrics
from sklearn.svm import LinearSVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn import linear_model

X = np.arange(-6, 6, 0.1)
#y = np.tanh(X)
y = np.sin(X)*np.tanh(X)+0.6*X-0.6

fig=plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.set_title('Hyperbolic Tangent Graph')
ax1.plot(X,y,label="original function")
ax2.plot(X,y,label="original function")
ax3.plot(X,y,label="original function")

e = [random.gauss(0, 0.3) for i in range(len(y))]
y += e

ax1.set_title('Hyperbolic Tangent Scatter')
ax1.scatter(X, y,label="original ")
ax3.set_title('Linear&Poly predict')
ax3.scatter(X, y,label="original ")

# 列ベクトルに変換する
X = X[:, np.newaxis]

# 学習を行う
#svr = svm.SVR(kernel='rbf')
#Lsvr = LinearSVR(random_state=0, tol=1e-5)
#lSGDR = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
KNR = KNeighborsRegressor(n_neighbors=5)
kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
GPR = GaussianProcessRegressor(kernel=kernel, alpha=0.0)

# RBFカーネル、線形、多項式でフィッティング
svr = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
Lsvr = svm.SVR(kernel='linear', C=1e3)
Psvr = svm.SVR(kernel='poly', C=1e3, degree=2)
lRidge = linear_model.Ridge(alpha=.5)
lBRidge = linear_model.BayesianRidge()

svr.fit(X, y)
KNR.fit(X, y)
GPR.fit(X, y)
Lsvr.fit(X, y)
Psvr.fit(X, y)
lRidge.fit(X, y)
lBRidge.fit(X, y)

# 回帰曲線を描く
X_plot = np.linspace(-10, 10, 10000)
y_svr = svr.predict(X_plot[:, np.newaxis])
y_KNR = KNR.predict(X_plot[:, np.newaxis])
y_GPR = GPR.predict(X_plot[:, np.newaxis])
y_Lsvr = Lsvr.predict(X_plot[:, np.newaxis])
y_Psvr = Psvr.predict(X_plot[:, np.newaxis])
y_lRidge = lRidge.predict(X_plot[:, np.newaxis])
y_lBRidge = lBRidge.predict(X_plot[:, np.newaxis])

#グラフにプロットする。
ax2.set_title('svm'+' predict')
ax2.scatter(X, y,label="original ")
ax2.plot(X_plot, y_svr,"r",label="SVR")
ax2.plot(X_plot, y_KNR,"b",label="KNR")
ax3.plot(X_plot, y_GPR,"g",label="GPR")
ax3.plot(X_plot, y_Lsvr,"r",label="SVR_Linear")
ax2.plot(X_plot, y_Psvr,"b",label="SVR_Poly")
ax3.plot(X_plot, y_lRidge,"g",label="Linear_Ridge")
ax3.plot(X_plot, y_lBRidge,"k",label="Bayesian_Ridge")

ax1.legend()  # 凡例をグラフにプロット
ax2.legend()  # 凡例をグラフにプロット
ax3.legend()  # 凡例をグラフにプロット

plt.savefig("sin(X)tanh(X)+06X-06_reg.jpg")
plt.show()
