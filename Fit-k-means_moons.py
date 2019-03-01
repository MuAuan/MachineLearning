import pandas as pd # データフレームワーク処理のライブラリをインポート
import matplotlib.pyplot as plt
from pandas.tools import plotting # 高度なプロットを行うツールのインポート
from sklearn.cluster import KMeans # K-means クラスタリングをおこなう
from sklearn.decomposition import PCA #主成分分析器
#from mpl_toolkits.mplot3d import Axes3D
#from PIL import Image
#from PIL import ImageOps
import numpy as np

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

"""    
X, y = make_blobs(n_samples=200,
                  n_features=3,
                  centers=8,
                  cluster_std=3.5,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1)  # For reproducibility
"""                  
x =X[:,0]  #df[df.columns[1]] #X[:,0]
y =X[:,1]  #df[df.columns[2]] #X[:,1]
print(x,y)

"""
df = pd.read_csv("keiba100_std.csv", sep=',', na_values=".") # データの読み込みSchoolScore.csv keiba2.csv
df.head() #データの確認
df.iloc[:, 1:].head() #解析に使うデータは２列目以降
#print(df[df.columns[1]])    ############ print(df[df.columns[1]]) #########
print(df.iloc[:, 1:])
X=df.iloc[:, 1:]


plotting.scatter_matrix(df[df.columns[1:]], figsize=(6,6), alpha=0.8, diagonal='kde')   #全体像を眺める
plt.savefig('k-means/keiba100/keiba_std2_scatter_plot4.jpg')
plt.pause(1)
plt.close()
"""

distortions = []
distortions1 = []
for i  in range(1,21):                # 1~20クラスタまで一気に計算 
    km = KMeans(n_clusters=i,
                init='k-means++',     # k-means++法によりクラスタ中心を選択
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)                # クラスタリングの計算を実行 #df.iloc[:, 1:]
    distortions.append(km.inertia_)       # km.fitするとkm.inertia_が得られる
    UF = km.inertia_ + i*np.log(20)*5e2   # km.inertia_ + kln(size)
    distortions1.append(UF)   
    
fig=plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
ax1.plot(range(1,21),distortions,marker='o')
ax1.set_xlabel('Number of clusters')
ax1.set_ylabel('Distortion')
#ax1.yscale('log')
ax2.plot(range(1,21),distortions,marker='o')
ax2.set_xlabel('Number of clusters')
ax2.set_ylabel('Distortion')
ax2.set_yscale('log')
ax3.plot(range(1,21),distortions1,marker='o')
ax3.set_xlabel('Number of clusters')
ax3.set_ylabel('Distortion+klog')
ax3.set_yscale('log')
plt.pause(1)
plt.savefig('k-means/moons/moons'+str(noise)+'_Distortion.jpg')
plt.close()

s=12
km = KMeans(n_clusters=s,
                init='k-means++',     # k-means++法によりクラスタ中心を選択
                n_init=10,
                max_iter=300,
                random_state=0)
y_km=km.fit_predict(X)  #df.iloc[:, 1:]

# この例では s個のグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
kmeans_model = KMeans(n_clusters=s, random_state=10).fit(X)  #df.iloc[:, 1:]
# 分類結果のラベルを取得する
labels = kmeans_model.labels_
print(labels)     ##################  print(labels) ###################
# それぞれに与える色を決める。
      
color_codes = {
    0:'#00FF00', 1:'#FF0000', 2:'#0000FF', 3:'#00FFFF' , 4:'#00FF77', 5:'#f0FF00', 6:'#FF06F0', 7:'#0070FF', 8:'#FFF444' , 9:'#077777',
    10:'#111111',11:'#222222',12:'#333333', 13:'#444444',14:'#555555',15:'#66FF00', 16:'#FF7777', 17:'#0000FF', 18:'#00FFFF',19:'#007777',
    20:'#f0FF00',21:'#FF0600',22:'#0070FF',23:'#08FFFF',24:'#577777',25:'#00FF00', 26:'#FF0000', 27:'#0000FF', 28:'#00FFFF',29:'#007777',
    30:'#177777',31:'#277777',32:'#377777',33:'#477777',34:'#577777',35:'#00FF00',36:'#FF0000',37:'#0000FF',38:'#00FFFF',39:'#007777',
    40:'#f0FF00',41:'#FF0600',42:'#0070FF',43:'#08FFFF',44:'#577777',45:'#00FF00', 46:'#FF0000', 47:'#0000FF', 48:'#00FFFF',49:'#007777',
}

# サンプル毎に色を与える。
colors = [color_codes[x] for x in labels]
#主成分分析の実行
pca = PCA()
pca.fit(X) #df.iloc[:, 1:]
PCA(copy=True, n_components=None, whiten=False)

# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(X) #df.iloc[:, 1:]
x,y = feature[:, 0], feature[:, 1]
print(x,y)                ##################  print(x,y); x,y = feature[:, 0], feature[:, 1]  ###################
X=np.c_[x,y]

plt.figure(figsize=(6, 6))
#for kx, ky, name in zip(x, y, df.iloc[:, 0]):
#    plt.text(kx, ky, name, alpha=0.8, size=10)
plt.scatter(x, y, alpha=0.8, color=colors)
#plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], c = "b", marker = "*", s = 50)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.savefig('k-means/moons/pca/moons'+str(noise)+'_PCA12_plotSn'+str(s)+'.jpg')
plt.pause(1)
plt.close()


from sklearn.metrics import silhouette_samples
from matplotlib import cm

cluster_labels = np.unique(y_km)       # y_kmの要素の中で重複を無くす s=6 ⇒ [0 1 2 3 4 5]
n_clusters=cluster_labels.shape[0]     # 配列の長さを返す。つまりここでは n_clustersで指定したsとなる

# シルエット係数を計算
silhouette_vals = silhouette_samples(X,y_km,metric='euclidean')  # サンプルデータ, クラスター番号、ユークリッド距離でシルエット係数計算 df.iloc[:, 1:]
y_ax_lower, y_ax_upper= 0,0
yticks = []

for i,c in enumerate(cluster_labels):
        c_silhouette_vals = silhouette_vals[y_km==c]      # cluster_labelsには 0,1,2が入っている（enumerateなのでiにも0,1,2が入ってる（たまたま））
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)              # サンプルの個数をクラスターごとに足し上げてy軸の最大値を決定
        color = cm.jet(float(i)/n_clusters)               # 色の値を作る
        plt.barh(range(y_ax_lower,y_ax_upper),            # 水平の棒グラフを描画（底辺の範囲を指定）
                         c_silhouette_vals,               # 棒の幅（1サンプルを表す）
                         height=1.0,                      # 棒の高さ
                         edgecolor='none',                # 棒の端の色
                         color=color)                     # 棒の色
        yticks.append((y_ax_lower+y_ax_upper)/2)          # クラスタラベルの表示位置を追加
        y_ax_lower += len(c_silhouette_vals)              # 底辺の値に棒の幅を追加

silhouette_avg = np.mean(silhouette_vals)                 # シルエット係数の平均値
plt.axvline(silhouette_avg,color="red",linestyle="--")    # 係数の平均値に破線を引く 
plt.yticks(yticks,cluster_labels + 1)                     # クラスタレベルを表示
plt.ylabel('Cluster')
plt.xlabel('silhouette coefficient')
plt.savefig('k-means/moons/moons'+str(noise)+'_silhouette_avg'+str(s)+'.jpg')
plt.pause(1)
plt.close()

# この例では s つのグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
kmeans_model = KMeans(n_clusters=s, random_state=10).fit(X)
# 分類結果のラベルを取得する
labels = np.unique(kmeans_model.labels_)  #kmeans_model.labels_ cluster_labels
#cluster_labels = np.unique(labels)   
# 分類結果を確認
print(labels)           #################   print(labels)   ############
# サンプル毎に色を与える。
colors = [color_codes[x] for x in kmeans_model.labels_]

kmeans_model = KMeans(n_clusters=s, random_state=10).fit(X)
print(kmeans_model.labels_)
print(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1])
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
for kx, ky, name in zip(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], labels):
    plt.text(kx, ky, name, alpha=0.8, size=20)
plt.scatter(x, y, alpha=0.8, color=colors)
plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], c = "b", marker = "*", s = 100)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.savefig('k-means/moons/pca/moons'+str(noise)+'_PCA12_plotn'+str(s)+'.jpg')
plt.pause(1)
plt.close()