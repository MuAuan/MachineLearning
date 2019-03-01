import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pandas.tools import plotting # 高度なプロットを行うツールのインポート
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

pca = PCA(n_components=2)
pca.fit(X)
v_pca0=pca.components_[0]
v_pca1=pca.components_[1]
X_pca0=[0,1*v_pca0[0]]
Y_pca0=[0,1*v_pca0[1]]
X_pca1=[0,1*v_pca1[0]]
Y_pca1=[0,1*v_pca1[1]]

plt.scatter(X[:,0],X[:,1],c="g",marker = "*")
plt.plot(X_pca0,Y_pca0,c="r",marker = "o")
plt.plot(X_pca1,Y_pca1,c="b",marker = "o")
plt.savefig('k-means/pca_example/pca_data_scatter.jpg')
plt.pause(1)
plt.close()
print(v_pca0[0],v_pca0[1])
Y=np.array([[0,0],[v_pca0[0],v_pca0[1]],[v_pca1[0],v_pca1[1]]])
print(Y)

pcatran_X = pca.transform(X)
pcatran_Y = pca.transform(Y)
print("pcatran_Y={}".format(pcatran_Y))

plt.scatter(pcatran_X[:,0],pcatran_X[:,1],c="g",marker = "*")
plt.scatter(pcatran_Y[:,0],pcatran_Y[:,1],c="r",marker = "o")
plt.plot([pcatran_Y[0,0],pcatran_Y[0,1]],[pcatran_Y[2,0],pcatran_Y[2,1]],c="b")
plt.plot([pcatran_Y[2,0],pcatran_Y[2,1]],[pcatran_Y[0,0],pcatran_Y[0,1]],c="r")  #,marker = "o")
plt.savefig('k-means/pca_example/pca_tran_scatter.jpg')
plt.pause(1)
plt.close()

print("v_pca0={},v_pca1={}".format(v_pca0,v_pca1))
s=2
print("pca.explained_variance_ratio_{}".format(pca.explained_variance_ratio_))
plt.bar([1, 2], pca.explained_variance_ratio_, align = "center",label="pca.explained_variance_ratio_={}".format(pca.explained_variance_ratio_))
plt.title("pca.explained_variance_ratio_")
plt.xlabel("components")
plt.ylabel("contribution")
plt.legend()
plt.savefig('k-means/pca_example/pca2d_contribution_PCA12_plotSn'+str(s)+'.jpg')
plt.pause(1)
plt.close()

print("pca.singular_values_={}".format(pca.singular_values_))  


#X = np.array([[-1, -1, -1], [-2, -1,-2], [-3, -2, 1], [1, 1,1], [2, 1,1], [3, 2,1],[-1, -1.5, -1], [-2, -1.5,-2], [-3, -2.5, 1], [1.5, 1,1], [2, 1.5,1], [3, 2,1.5],[-1, 0, -1], [0, -1,-2], [-3, -2, 0], [1, 0,1], [2, 0,1], [0, 2,1],[-1, -0.5, -1], [0, -1.5,-2], [0, -2.5, 1], [1.5, 0,1], [2, 0.5,1], [3, 2,0.5]])
from sklearn.datasets import make_regression

#X, Y, coef = make_regression(random_state=12, n_samples=100, n_features=3, n_informative=2, noise=10.0, bias=-0.0, coef=True)

from sklearn.datasets import make_blobs
X, Y = make_blobs(random_state=8,
                  n_samples=100, 
                  n_features=3, 
                  cluster_std=4,
                  centers=3)

pca = PCA(n_components=3)
pca.fit(X)
v_pca0=pca.components_[0]
v_pca1=pca.components_[1]
v_pca2=pca.components_[2]
X_pca0=[0,10*v_pca0[0]]
Y_pca0=[0,10*v_pca0[1]]
Z_pca0=[0,10*v_pca0[2]]
X_pca1=[0,10*v_pca1[0]]
Y_pca1=[0,10*v_pca1[1]]
Z_pca1=[0,10*v_pca1[2]]
X_pca2=[0,10*v_pca2[0]]
Y_pca2=[0,10*v_pca2[1]]
Z_pca2=[0,10*v_pca2[2]]

def show_with_angle(angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = 10., azim = angle)
    ax.plot(X_pca0,Y_pca0,Z_pca0, c='b', marker='o', alpha=0.5, label='1st pricipal')  
    ax.plot(X_pca1,Y_pca1,Z_pca1, c='r', marker='o', alpha=0.5, label='2nd principal') 
    ax.plot(X_pca2,Y_pca2,Z_pca2, c='g', marker='o', alpha=0.5, label='3rd principal') 
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],  c="b", marker='o', alpha=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.pause(0.01)
    plt.savefig('k-means/pca_example/pca3d/keiba3_PCA3d_angle_'+str(angle)+'.jpg')
    plt.close()
 
for angle in range(0, 360, 5):
    show_with_angle(angle)

print(v_pca0[0],v_pca0[1],v_pca0[2])
Y=np.array([[0,0,0],[10*v_pca0[0],10*v_pca0[1],10*v_pca0[2]],[10*v_pca1[0],10*v_pca1[1],10*v_pca1[2]],[10*v_pca2[0],10*v_pca2[1],10*v_pca2[2]]])
print(Y)

pcatran_X = pca.transform(X)
print("pcatran_X={}".format(pcatran_X))
pcatran_Y = pca.transform(Y)
print("pcatran_Y={}".format(pcatran_Y))

plt.scatter(pcatran_X[:,0],pcatran_X[:,1],c="g",marker = "*")
plt.scatter(pcatran_Y[:,0],pcatran_Y[:,1],c="r",marker = "o")
plt.plot([pcatran_Y[0,0],pcatran_Y[1,0]],[pcatran_Y[0,1],pcatran_Y[1,1]],c="b",marker = "o")
plt.plot([pcatran_Y[0,0],pcatran_Y[2,0]],[pcatran_Y[0,1],pcatran_Y[2,1]],c="r",marker = "o")
plt.savefig('k-means/pca_example/pca_tran_scatter3.jpg')
plt.pause(1)
plt.close()

def show_with_angle(angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = 10., azim = angle)
    ax.scatter(pcatran_X[:, 0], pcatran_X[:, 1], pcatran_X[:, 2],  c="b", marker='o', alpha=1)
    ax.set_xlabel('tran_x')
    ax.set_ylabel('tran_y')
    ax.set_zlabel('tran_z')
    ax.legend()
    plt.pause(0.01)
    plt.savefig('k-means/pca_example/tran_pca3d/pca3_PCA3d_angle_'+str(angle)+'.jpg')
    plt.close()
 
for angle in range(0, 360, 5):
    show_with_angle(angle)

print("v_pca0={},v_pca1={},v_pca2={}".format(v_pca0,v_pca1,v_pca2))
s=3
print("pca.explained_variance_ratio_{}".format(pca.explained_variance_ratio_))
plt.bar([1, 2,3], pca.explained_variance_ratio_, align = "center",label="pca.explained_variance_ratio_={}".format(pca.explained_variance_ratio_))
plt.title("pca.explained_variance_ratio_")
plt.xlabel("components")
plt.ylabel("contribution")
plt.legend()
plt.savefig('k-means/pca_example/pca3d_contribution_PCA12_plotSn'+str(s)+'.jpg')
plt.pause(1)
plt.close()

print("pca.singular_values_={}".format(pca.singular_values_))  

s=6
#from sklearn.datasets import make_blobs
X, Y = make_blobs(random_state=8,
                  n_samples=100, 
                  n_features=6, 
                  cluster_std=4,
                  centers=3)


pca = PCA(n_components=6)
pca.fit(X)
v_pca0=pca.components_[0]
v_pca1=pca.components_[1]
v_pca2=pca.components_[2]

X_pca0=[0,10*v_pca0[0]]
Y_pca0=[0,10*v_pca0[1]]
Z_pca0=[0,10*v_pca0[2]]
X_pca1=[0,10*v_pca1[0]]
Y_pca1=[0,10*v_pca1[1]]
Z_pca1=[0,10*v_pca1[2]]
X_pca2=[0,10*v_pca2[0]]
Y_pca2=[0,10*v_pca2[1]]
Z_pca2=[0,10*v_pca2[2]]

def show_with_angle(angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = 10., azim = angle)
    ax.plot(X_pca0,Y_pca0,Z_pca0, c='b', marker='o', alpha=0.5, label='1st pricipal')  
    ax.plot(X_pca1,Y_pca1,Z_pca1, c='r', marker='o', alpha=0.5, label='2nd principal') 
    ax.plot(X_pca2,Y_pca2,Z_pca2, c='g', marker='o', alpha=0.5, label='3rd principal') 
    ax.scatter(X[:, 0], X[:, 1], X[:, 2],  c="b", marker='o', alpha=1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.pause(0.01)
    plt.savefig('k-means/pca_example/pca6d/keiba3_PCA3d_angle_'+str(angle)+'.jpg')
    plt.close()
 
for angle in range(0, 360, 5):
    show_with_angle(angle)

print(v_pca0[0],v_pca0[1],v_pca0[2])

pcatran_X = pca.transform(X)
print("pcatran_X={}".format(pcatran_X))

def show_with_angle(angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = 10., azim = angle)
    ax.scatter(pcatran_X[:, 0], pcatran_X[:, 1], pcatran_X[:, 2],  c="b", marker='o', alpha=1)
    ax.set_xlabel('tran_x')
    ax.set_ylabel('tran_y')
    ax.set_zlabel('tran_z')
    ax.legend()
    plt.pause(0.01)
    plt.savefig('k-means/pca_example/tran_pca6d/pca3_PCA3d_angle_'+str(angle)+'.jpg')
    plt.close()
 
for angle in range(0, 360, 5):
    show_with_angle(angle)

print("v_pca0={},v_pca1={},v_pca2={}".format(v_pca0,v_pca1,v_pca2))
s=3
print("pca.explained_variance_ratio_{}".format(pca.explained_variance_ratio_))
plt.bar([1, 2,3,4,5,6], pca.explained_variance_ratio_, align = "center",label="pca.explained_variance_ratio_={}".format(pca.explained_variance_ratio_))
plt.title("pca.explained_variance_ratio_")
plt.xlabel("components")
plt.ylabel("contribution")
plt.legend()
plt.savefig('k-means/pca_example/pca6d_contribution_PCA12_plotSn'+str(s)+'.jpg')
plt.pause(1)
plt.close()

print("pca.singular_values_={}".format(pca.singular_values_))  