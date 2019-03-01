# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from IPython.display import display, HTML # Jupyter notebook用

class XMeans:
    """
    x-means法を行うクラス
    """

    def __init__(self, k_init = 2, **k_means_args): #k_init = 2
        """
        k_init : The initial number of clusters applied to KMeans()
        """
        self.k_init = k_init
        self.k_means_args = k_means_args

    def fit(self, X):
        """
        x-means法を使ってデータXをクラスタリングする
        X : array-like or sparse matrix, shape=(n_samples, n_features)
        """
        self.__clusters = [] 

        clusters = self.Cluster.build(X, KMeans(self.k_init, **self.k_means_args).fit(X))
        self.__recursively_split(clusters)

        self.labels_ = np.empty(X.shape[0], dtype = np.intp)
        for i, c in enumerate(self.__clusters):
            self.labels_[c.index] = i

        self.cluster_centers_ = np.array([c.center for c in self.__clusters])
        self.cluster_log_likelihoods_ = np.array([c.log_likelihood() for c in self.__clusters])
        self.cluster_sizes_ = np.array([c.size for c in self.__clusters])

        return self

    def __recursively_split(self, clusters):
        """
        引数のclustersを再帰的に分割する
        clusters : list-like object, which contains instances of 'XMeans.Cluster'
        'XMeans.Cluster'のインスタンスを含むリスト型オブジェクト
        """
        for cluster in clusters:
            if cluster.size <= 3:  #3
                self.__clusters.append(cluster)
                continue

            k_means = KMeans(2, **self.k_means_args).fit(cluster.data)
            c1, c2 = self.Cluster.build(cluster.data, k_means, cluster.index)

            beta = np.linalg.norm(c1.center - c2.center) / np.sqrt(np.linalg.det(c1.cov) + np.linalg.det(c2.cov))
            alpha = 0.5 / stats.norm.cdf(beta)
            bic = -2 * (cluster.size * np.log(alpha) + c1.log_likelihood() + c2.log_likelihood()) + 2 * cluster.df * np.log(cluster.size)

            if bic < cluster.bic():
                self.__recursively_split([c1, c2])
            else:
                self.__clusters.append(cluster)
    
    
    class Cluster:
        """
        k-means法によって生成されたクラスタに関する情報を持ち、尤度やBICの計算を行うクラス
        """

        @classmethod
        def build(cls, X, k_means, index = "None"): 
            if index == "None":
                index = np.array(range(0, X.shape[0]))
            labels = range(0, k_means.get_params()["n_clusters"])
            print(labels)

            return tuple(cls(X, index, k_means, label) for label in labels) 

        # index: Xの各行におけるサンプルが元データの何行目のものかを示すベクトル
        def __init__(self, X, index, k_means, label):
            self.data = X[k_means.labels_ == label]   #ok 10 ng 3 5 8
            #print('self.data=',self.data)
            self.index = index[k_means.labels_ == label]
            self.size = self.data.shape[0]
            self.df = self.data.shape[1] * (self.data.shape[1] + 3) / 2
            self.center = k_means.cluster_centers_[label]
            self.cov = np.cov(self.data.T)
            #print('self.cov=',self.cov)

        def log_likelihood(self):
            #print(self.cov)
            #print(np.linalg.matrix_rank(self.cov)) 
            return sum(stats.multivariate_normal.logpdf(x, self.center, self.cov) for x in self.data)

        def bic(self):
            return -2 * self.log_likelihood() + self.df * np.log(self.size)
            
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # データの準備
    #x = np.array([np.random.normal(loc, 0.1, 15) for loc in np.repeat([1,2], 2)]).flatten() #ランダムな80個の数を生成 20
    #y = np.array([np.random.normal(loc, 0.1, 15) for loc in np.tile([1,2], 2)]).flatten() #ランダムな80個の数を生成 20

    from sklearn.datasets import make_blobs
    from sklearn.datasets import make_moons

    N =300
    noise=0.05
    X, y = make_moons(N,
                  noise=noise,
                 random_state=0)
    plt.scatter(X[:,0],X[:,1])
    plt.pause(1)
    plt.close()

    x =X[:,0]  #df[df.columns[1]] #X[:,0]
    y =X[:,1]  #df[df.columns[2]] #X[:,1]
    
    """
    X, y = make_blobs(n_samples=200,
                      n_features=3,
                      centers=8,
                      cluster_std=3.5,
                      center_box=(-10.0, 10.0),
                      shuffle=True,
                      random_state=1)  # For reproducibility
    x =X[:,0]  #df[df.columns[1]] #X[:,0]
    y =X[:,1]  #df[df.columns[2]] #X[:,1]

    
    import pandas as pd # データフレームワーク処理のライブラリをインポート
    df = pd.read_csv("keiba100_std.csv", sep=',', na_values=".") # データの読み込みSchoolScore.csv keiba2.csv keiba100_std.csv
    df.head() #データの確認
    df.iloc[:, 1:].head() #解析に使うデータは２列目以降
    print(df)
    X=df.iloc[:, 1:]
    """
    from sklearn.decomposition import PCA #主成分分析器
    #主成分分析の実行
    pca = PCA()
    pca.fit(X)  #df.iloc[:, 1:]
    PCA(copy=True, n_components=None, whiten=False)
    v_pca0 = pca.components_[0]
    v_pca1 = pca.components_[1]
    #v_pca2 = pca.components_[2]
    
    print("v_pca0={},v_pca1={}".format(v_pca0,v_pca1))
    
    from numpy.random import *

    # データを主成分空間に写像 = 次元圧縮
    feature = pca.transform(X)  #df.iloc[:, 1:]
    x,y = feature[:, 0], feature[:, 1]
    print(x,y)
    
    X=np.c_[x,y]  #df.iloc[:, 1:]  #np.c_[x,y]
    
    # クラスタリングの実行
    # この例では 3 つのグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
    kmeans_model = KMeans(n_clusters=3, random_state=10).fit(X)  #X df.iloc[:, 1:]
    # 分類結果のラベルを取得する
    labels = kmeans_model.labels_
    # 分類結果を確認
    print(labels)
    
    # それぞれに与える色を決める。
    color_codes = {0:'#00FF00', 1:'#FF0000', 2:'#0000FF', 3:'#00FFFF' , 4:'#007777', 5:'#f0FF00', 6:'#FF0600', 7:'#0070FF', 8:'#08FFFF' , 9:'#077777',10:'#177777', 11:'#277777', 12:'#377777', 13:'#477777' , 14:'#577777'}
    # サンプル毎に色を与える。
    colors = [color_codes[x] for x in labels]
    plt.scatter(x,y,c=colors,marker='o',s=50)
    print(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1])
    plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], c = colors, marker = "*", s = 100)
    plt.title("x-means_test1")
    plt.grid()
    plt.savefig("./k-means/keiba100/data_clustering0_10.png", dpi = 200)     
    plt.show()
    
    x_means = XMeans(random_state = 10).fit(np.c_[X])   #df.iloc[:, 1:])  #np.c_[X])    #np.c_[x,y])  
    print(x_means.labels_)
    print(x_means.cluster_centers_)
    print(x_means.cluster_log_likelihoods_)
    print(x_means.cluster_sizes_)
    colors = [color_codes[x] for x in x_means.labels_]
    
    # 結果をプロット
    plt.rcParams["font.family"] = "Hiragino Kaku Gothic Pro"
    plt.scatter(x, y, c = colors, s = 30)  #x_means.labels_
    plt.scatter(x_means.cluster_centers_[:,0], x_means.cluster_centers_[:,1], c = "r", marker = "*", s = 250)  #k_means.cluster_centers_
    #plt.xlim(0, 3)
    #plt.ylim(0, 3)
    plt.title("x-means_test1")
    #plt.legend()
    plt.grid()
    plt.savefig("./k-means/keiba100/data_clustering_10.png", dpi = 200)      
    plt.show()    