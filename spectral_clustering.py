from sklearn.cluster import SpectralClustering
import numpy as np
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import cluster, preprocessing #機械学習用のライブラリを利用
from sklearn import datasets #使用するデータ
from sklearn.datasets import make_moons

#X = np.array([[1, 1], [2, 1], [1, 0], [4, 7], [3, 5], [3, 6]])
# 2：moon型のデータを読み込む--------------------------------
#X,z = datasets.make_moons(n_samples=200, noise=0.05, random_state=0)

N =300
noise=0.05
X, z = make_moons(N,
                  noise=noise,
                 random_state=0)
plt.scatter(X[:,0],X[:,1])
plt.title("original")
plt.pause(1)
plt.close()


clustering = SpectralClustering(n_clusters=2,
        assign_labels="discretize",
        random_state=0,affinity="nearest_neighbors").fit(X)
print(clustering.labels_)

plt.scatter(X[:,0],X[:,1], c=clustering.labels_)
plt.title("discretize")
plt.pause(1)
plt.close()

clustering = SpectralClustering(n_clusters=2,
        assign_labels="kmeans",
        random_state=0,affinity="nearest_neighbors").fit(X)
print(clustering.labels_)

plt.scatter(X[:,0],X[:,1], c=clustering.labels_)
plt.title("kmeans")
plt.pause(1)
plt.close()


clustering=cluster.SpectralClustering(n_clusters=2, affinity="nearest_neighbors").fit(X) #affinity='precomputed'
print(clustering.labels_)

plt.scatter(X[:,0],X[:,1], c=clustering.labels_)
plt.title("nearest_neighbors")
plt.pause(1)
plt.close()

clustering=cluster.SpectralClustering(n_clusters=2, affinity="precomputed").fit(X) #affinity='precomputed'
print(clustering.labels_)

plt.scatter(X[:,0],X[:,1], c=clustering.labels_)
plt.title("nearest_neighbors")
plt.pause(1)
plt.close()