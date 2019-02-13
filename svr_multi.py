import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

m = 2 #dimension
mean = np.zeros(m)
sigma = np.eye(m)

N = 100
x1 = np.linspace(-5, 5, N)
x2 = np.linspace(-5, 5, N)

X1, X2 = np.meshgrid(x1, x2)
X = np.c_[np.ravel(X1), np.ravel(X2)]

# アウトプットを算出
y = np.sin(X1).ravel()+ np.cos(X2).ravel()
y_o = y.copy()
y1 = y.reshape(X1.shape)

#Y_plot = multivariate_normal.pdf(x=X, mean=mean, cov=sigma)
#Y_plot = Y_plot.reshape(X1.shape)
# ノイズを加える
y1[::5] += 1 * (0.5 - np.random.rand(20,100))

fig=plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')
surf = ax2.plot_surface(X1, X2, y1, cmap='bwr', linewidth=0) #Y_plot
fig.colorbar(surf)
ax1.scatter3D(np.ravel(X1), np.ravel(X2), y)
ax2.set_title("Surface Plot")
ax1.set_title("scatter Plot")
plt.savefig('Surface_plot2_input.jpg')
plt.pause(1)
plt.close()

# フィッティング
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
#svr_lin = SVR(kernel='linear', C=1e3)
#svr_poly = SVR(kernel='poly', C=1e3, degree=3)
y_rbf = svr_rbf.fit(X, y).predict(X)
#y_lin = svr_lin.fit(X, y).predict(X)
#y_poly = svr_poly.fit(X, y).predict(X)

# テストデータも準備
#test_X1 = np.sort(5 * np.random.rand(40, 1).reshape(40), axis=0)
#test_X2 = np.sort(3 * np.random.rand(40, 1).reshape(40), axis=0)
N = 100
test_x1 = np.linspace(-5, 5, N)
test_x2 = np.linspace(-5, 5, N)

test_X1, test_X2 = np.meshgrid(test_x1, test_x2)
test_X = np.c_[np.ravel(test_X1), np.ravel(test_X2)]

#test_X = np.c_[test_X1, test_X2]
test_y = np.sin(test_X1).ravel() + np.cos(test_X2).ravel()

# テストデータを突っ込んで推定してみる
test_rbf = svr_rbf.predict(test_X)
#test_lin = svr_lin.predict(test_X)
#test_poly = svr_poly.predict(test_X)
test_rbf1 = test_rbf.reshape(test_X1.shape)

fig=plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(211, projection='3d')
ax2 = fig.add_subplot(212, projection='3d')
#ax3 = fig.add_subplot(313)
ax1.set_title('Graph')
ax1.scatter3D(np.ravel(test_X1), np.ravel(test_X2), test_rbf)
surf = ax2.plot_surface(test_X1, test_X2, test_rbf1, cmap='bwr', linewidth=0) #Y_plot
fig.colorbar(surf)
#ax1.scatter3D(np.ravel(test_X1), np.ravel(test_X2), test_lin)
#ax1.scatter3D(np.ravel(test_X1), np.ravel(test_X2), test_poly)
plt.savefig('Surface_svr_rbf_prediction.jpg')
plt.pause(1)
plt.close()

from sklearn.metrics import mean_squared_error
from math import sqrt

# 相関係数計算
rbf_corr = np.corrcoef(test_y, test_rbf)[0, 1]
#lin_corr = np.corrcoef(test_y, test_lin)[0, 1]
#poly_corr = np.corrcoef(test_y, test_poly)[0, 1]

# RMSEを計算
rbf_rmse = sqrt(mean_squared_error(test_y, test_rbf))
#lin_rmse = sqrt(mean_squared_error(test_y, test_lin))
#poly_rmse = sqrt(mean_squared_error(test_y, test_poly))

print( "RBF: RMSE %f \t\t Corr %f" % (rbf_rmse, rbf_corr))
#print( "Linear: RMSE %f \t Corr %f" % (lin_rmse, lin_corr))
#print( "Poly: RMSE %f \t\t Corr %f" % (poly_rmse, poly_corr))
