import pandas as pd # データフレームワーク処理のライブラリをインポート
import matplotlib.pyplot as plt
from pandas.tools import plotting # 高度なプロットを行うツールのインポート
from sklearn.cluster import KMeans # K-means クラスタリングをおこなう
from sklearn.decomposition import PCA #主成分分析器
#from mpl_toolkits.mplot3d import Axes3D
#from PIL import Image
#from PIL import ImageOps
import numpy as np


df = pd.read_csv("keiba100_std.csv", sep=',', na_values=".") # データの読み込みSchoolScore.csv keiba2.csv
df.head() #データの確認
df.iloc[:, 1:].head() #解析に使うデータは２列目以降
#print(df[df.columns[1]])    ############ print(df[df.columns[1]]) #########
print(df.iloc[:, 1:])

plotting.scatter_matrix(df[df.columns[1:]], figsize=(6,6), alpha=0.8, diagonal='kde')   #全体像を眺める
plt.savefig('k-means/keiba100/keiba_std2_scatter_plot4.jpg')
plt.pause(1)
plt.close()

distortions = []
distortions1 = []
for i  in range(1,21):                # 1~20クラスタまで一気に計算 
    km = KMeans(n_clusters=i,
                init='k-means++',     # k-means++法によりクラスタ中心を選択
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(df.iloc[:, 1:])                # クラスタリングの計算を実行
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
plt.savefig('k-means/keiba100/keiba_std2_Distortion.jpg')
plt.close()

s=15
km = KMeans(n_clusters=s,
                init='k-means++',     # k-means++法によりクラスタ中心を選択
                n_init=10,
                max_iter=300,
                random_state=0)
y_km=km.fit_predict(df.iloc[:, 1:])

# この例では s個のグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
kmeans_model = KMeans(n_clusters=s, random_state=10).fit(df.iloc[:, 1:])

# 分類結果のラベルを取得する
labels = kmeans_model.labels_
print(labels)     ##################  print(labels) ###################
# それぞれに与える色を決める。
color_codes = {0:'#00FF00', 1:'#FF0000', 2:'#0000FF', 3:'#00FFFF' , 4:'#007777', 5:'#f0FF00', 6:'#FF0600', 7:'#0070FF', 8:'#08FFFF' , 9:'#077777',10:'#177777', 11:'#277777', 12:'#377777', 13:'#477777' , 14:'#577777'}
# サンプル毎に色を与える。
colors = [color_codes[x] for x in labels]
#主成分分析の実行
pca = PCA()
pca.fit(df.iloc[:, 1:])
PCA(copy=True, n_components=None, whiten=False)
v_pca0 = pca.components_[0]
v_pca1 = pca.components_[1]
v_pca2 = pca.components_[2]
    
print("v_pca0={},v_pca1={},v_pca2={}".format(v_pca0,v_pca1,v_pca2))

# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(df.iloc[:, 1:])
x,y = feature[:, 0], feature[:, 1]
print(x,y,feature[:, 2])                ##################  print(x,y); x,y = feature[:, 0], feature[:, 1]  ###################
X=np.c_[x,y]

plt.figure(figsize=(6, 6))
for kx, ky, name in zip(x, y, df.iloc[:, 0]):
    plt.text(kx, ky, name, alpha=0.8, size=10)
plt.scatter(x, y, alpha=0.8, color=colors)
#plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], c = "b", marker = "*", s = 50)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.savefig('k-means/keiba100/pca/keiba_std2_PCA12_plotSn'+str(s)+'.jpg')
plt.pause(1)
plt.close()


from sklearn.metrics import silhouette_samples
from matplotlib import cm

cluster_labels = np.unique(y_km)       # y_kmの要素の中で重複を無くす s=6 ⇒ [0 1 2 3 4 5]
n_clusters=cluster_labels.shape[0]     # 配列の長さを返す。つまりここでは n_clustersで指定したsとなる

# シルエット係数を計算
silhouette_vals = silhouette_samples(df.iloc[:, 1:],y_km,metric='euclidean')  # サンプルデータ, クラスター番号、ユークリッド距離でシルエット係数計算
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
plt.savefig('k-means/keiba100/keiba_std2_silhouette_avg'+str(s)+'.jpg')
plt.pause(1)
plt.close()

# この例では 3 つのグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
#kmeans_model = KMeans(n_clusters=s, random_state=10).fit(df.iloc[:, 1:])
# 分類結果のラベルを取得する
labels = cluster_labels  #kmeans_model.labels_
# 分類結果を確認
print(labels)           #################   print(labels)   ############
# サンプル毎に色を与える。
colors = [color_codes[x] for x in labels]

# 色分けした Scatter Matrix を描く。
plotting.scatter_matrix(df[df.columns[1:]], figsize=(6,6),c=colors, diagonal='kde', alpha=0.8)   #データのプロット
plt.savefig('k-means/keiba100/keiba_std2_scatter_color_plot'+str(s)+'.jpg')
plt.pause(1)
plt.close()
"""
import seaborn as sns
pg = sns.pairplot(df)

print(type(pg))
pg.savefig('k-means/seaborn_pairplot_default.png')
"""
fig, axes = plt.subplots(nrows=13, ncols=13, figsize=(18, 18),sharex=False,sharey=False)
for i in range(1,14):
    for j in range(1,14):
        ax = axes[i-1,j-1]
        x_data=df[df.columns[i]]
        y_data=df[df.columns[j]]
        #print(x_data)
        ax.scatter(x_data,y_data,c=colors)
        cor=np.corrcoef(x_data,y_data)[0, 1]
        ax.set_title("{:.3f}".format(cor))
       

fig.savefig('k-means/keiba100/pandas_std2_iris_line_axes'+str(s)+'.png')

"""
#主成分分析の実行
pca = PCA()
pca.fit(df.iloc[:, 1:])
PCA(copy=True, n_components=None, whiten=False)

# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(df.iloc[:, 1:])
x,y = feature[:, 0], feature[:, 1]
print(x,y)
X=np.c_[x,y]
#print(df.iloc[:, 0])
"""
kmeans_model = KMeans(n_clusters=s, random_state=10).fit(X)
print(kmeans_model.labels_)
print(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1])
v_pca0 = pca.components_[0]
v_pca1 = pca.components_[1]
v_pca2 = pca.components_[2]
print("v_pca0={},v_pca1={},v_pca2={}".format(v_pca0,v_pca1,v_pca2))

colors = [color_codes[x] for x in kmeans_model.labels_]
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
for kx, ky, name in zip(x, y, df.iloc[:, 0]):
    plt.text(kx, ky, name, alpha=0.8, size=10)
plt.scatter(x, y, alpha=0.8, color=colors)
plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], c = "b", marker = "*", s = 100)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.savefig('k-means/keiba100/pca/keiba_std2_PCA12_plotn'+str(s)+'.jpg')
plt.pause(1)
plt.close()