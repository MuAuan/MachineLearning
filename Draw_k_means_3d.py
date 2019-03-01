import pandas as pd # データフレームワーク処理のライブラリをインポート
import matplotlib.pyplot as plt
from pandas.tools import plotting # 高度なプロットを行うツールのインポート
from sklearn.cluster import KMeans # K-means クラスタリングをおこなう
from sklearn.decomposition import PCA #主成分分析器
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


df = pd.read_csv("keiba100_std.csv", sep=',', na_values=".") # データの読み込み
df.iloc[:, 1:].head() #解析に使うデータは２列目以降
print(df.iloc[:, 1:])
X=df.iloc[:, 1:]

plotting.scatter_matrix(df[df.columns[1:]], figsize=(6,6), alpha=0.8, diagonal='kde')   #全体像を眺める
plt.savefig('k-means/keiba100/keiba3_scatter_plot4.jpg')
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
ax2.plot(range(1,21),distortions,marker='o')
ax2.set_xlabel('Number of clusters')
ax2.set_ylabel('Distortion')
ax2.set_yscale('log')
ax3.plot(range(1,21),distortions1,marker='o')
ax3.set_xlabel('Number of clusters')
ax3.set_ylabel('Distortion+klog')
ax3.set_yscale('log')
plt.pause(1)
plt.savefig('k-means/keiba100/keiba3__Distortion.jpg')
plt.close()

s=3
# この例では s個のグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
kmeans_model = KMeans(n_clusters=s, random_state=10).fit(df.iloc[:, 1:])

# 分類結果のラベルを取得する
labels = kmeans_model.labels_
print(labels)     ##################  print(labels) ###################
# それぞれに与える色を決める。
color_codes = {0:'#00FF00', 1:'#FF0000', 2:'#0000FF', 3:'#00FFFF' , 4:'#007777', 5:'#f0FF00', 6:'#FF0600', 7:'#0070FF', 8:'#08FFFF' , 9:'#077777',10:'#177777', 11:'#277777', 12:'#377777', 13:'#477777' , 14:'#577777'}
# サンプル毎に色を与える。
colors = [color_codes[x] for x in labels]

fig, axes = plt.subplots(nrows=13, ncols=13, figsize=(18, 18),sharex=False,sharey=False)
for i in range(1,14):
    for j in range(1,14):
        ax = axes[i-1,j-1]
        x_data=df[df.columns[i]]
        y_data=df[df.columns[j]]
        ax.scatter(x_data,y_data,c=colors)
        cor=np.corrcoef(x_data,y_data)[0, 1]
        ax.set_title("{:.3f}".format(cor),fontsize=30)
plt.savefig('k-means/keiba100/keiba3_correlation.jpg')
plt.close()

#主成分分析の実行
pca = PCA()
pca.fit(df.iloc[:, 1:])
PCA(copy=True, n_components=None, whiten=False)
v_pca0 = pca.components_[0]
v_pca1 = pca.components_[1]
v_pca2 = pca.components_[2]

print("v_pca0={},v_pca1={},v_pca2={}".format(v_pca0,v_pca1,v_pca2))
X_pca1=[50,50+30*v_pca0[0]]
Y_pca1=[50,50+30*v_pca0[1]]
Z_pca1=[50,50+30*v_pca0[2]]
X_pca2=[50,50+90*v_pca1[0]]
Y_pca2=[50,50+90*v_pca1[1]]
Z_pca2=[50,50+90*v_pca1[2]]
X_pca3=[50,50+90*v_pca2[0]]
Y_pca3=[50,50+90*v_pca2[1]]
Z_pca3=[50,50+90*v_pca2[2]]

# 分類結果のラベルを取得する
labels = np.unique(kmeans_model.labels_)

def show_with_angle(angle):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev = 10., azim = angle)
    for kx, ky,kz, name in zip(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], kmeans_model.cluster_centers_[:,2], labels):
        ax.text(kx, ky,kz, name, alpha=0.8, size=20)
    ax.plot(X_pca1,Y_pca1,Z_pca1, c='b', marker='o', alpha=0.5, label='1st pricipal')  
    ax.plot(X_pca2,Y_pca2,Z_pca2, c='r', marker='o', alpha=0.5, label='2nd principal') 
    ax.plot(X_pca3,Y_pca3,Z_pca3, c='g', marker='o', alpha=0.5, label='3rd principal') 
    ax.scatter(df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3],  c=colors, marker='o', alpha=1)
    ax.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], kmeans_model.cluster_centers_[:,2], c = "b", marker = "*", s = 100)
    ax.set_xlabel('1st')
    ax.set_ylabel('2nd')
    ax.set_zlabel('3rd')
    ax.legend()
    plt.pause(0.01)
    plt.savefig('k-means/keiba100/pca3d/keiba3_PCA3d_angle_'+str(angle)+'.jpg')
    plt.close()
 
for angle in range(0, 360, 1):
    show_with_angle(angle)

print("pca.explained_variance_ratio_")
plt.bar([1, 2,3,4,5,6,7,8,9,10,11,12,13], pca.explained_variance_ratio_, align = "center",label="pca.explained_variance_ratio_={}".format(pca.explained_variance_ratio_))
plt.title("pca.explained_variance_ratio_")
plt.xlabel("components")
plt.ylabel("contribution")
plt.legend()
plt.savefig('k-means/keiba100/keiba3_contribution_PCA12_plotSn'+str(s)+'.jpg')
plt.pause(1)
plt.close()

# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(df.iloc[:, 1:])
x,y = feature[:, 0], feature[:, 1]
print(x,y)                ##################  print(x,y); x,y = feature[:, 0], feature[:, 1]  ###################
X=np.c_[x,y]

kmeans_model = KMeans(n_clusters=s, random_state=10).fit(X) #PCA平面（ｘ、ｙ）で再度クラスタリング

plt.figure(figsize=(6, 6))
for kx, ky, name in zip(x, y, df.iloc[:, 0]):
    plt.text(kx, ky, name, alpha=0.8, size=10)
plt.scatter(x, y, alpha=0.8, color=colors)
plt.scatter(kmeans_model.cluster_centers_[:,0], kmeans_model.cluster_centers_[:,1], c ="b" , marker = "*", s = 200)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.savefig('k-means/keiba100/keiba3_std2_PCA12_plotSn'+str(s)+'.jpg')
plt.pause(1)
plt.close()
