# URL によるリソースへのアクセスを提供するライブラリをインポートする。
import urllib.request
import pandas as pd # データフレームワーク処理のライブラリをインポート
# 図やグラフを図示するためのライブラリをインポートする。
import matplotlib.pyplot as plt
from pandas.tools import plotting # 高度なプロットを行うツールのインポート
from sklearn.cluster import KMeans # K-means クラスタリングをおこなう
#import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器


# ウェブ上のリソースを指定する
url = 'https://raw.githubusercontent.com/maskot1977/ipython_notebook/master/toydata/SchoolScore.txt'
# 指定したURLからリソースをダウンロードし、名前をつける。
urllib.request.urlretrieve(url, 'SchoolScore.txt')
#('SchoolScore.txt', <httplib.HTTPMessage instance at 0x104143e18>)

df = pd.read_csv("SchoolScore.txt", sep='\t', na_values=".") # データの読み込み
df.head() #データの確認
df.iloc[:, 1:].head() #解析に使うデータは２列目以降

plotting.scatter_matrix(df[df.columns[1:]], figsize=(6,6), alpha=0.8, diagonal='kde')   #全体像を眺める
plt.savefig('k-means/scatter_plot.jpg')
plt.pause(1)
plt.close()

# この例では 3 つのグループに分割 (メルセンヌツイスターの乱数の種を 10 とする)
kmeans_model = KMeans(n_clusters=3, random_state=10).fit(df.iloc[:, 1:])
# 分類結果のラベルを取得する
labels = kmeans_model.labels_

# 分類結果を確認
print(labels)
#array([2, 2, 0, 2, 2, 1, 2, 2, 1, 2, 2, 2, 1, 2, 2, 0, 1, 2, 0, 0, 1, 0, 1,
#       1, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1], dtype=int32)

# それぞれに与える色を決める。
color_codes = {0:'#00FF00', 1:'#FF0000', 2:'#0000FF'}
# サンプル毎に色を与える。
colors = [color_codes[x] for x in labels]

# 色分けした Scatter Matrix を描く。
plotting.scatter_matrix(df[df.columns[1:]], figsize=(6,6),c=colors, diagonal='kde', alpha=0.8)   #データのプロット
plt.savefig('k-means/scatter_color_plot.jpg')
plt.pause(1)
plt.close()

#主成分分析の実行
pca = PCA()
pca.fit(df.iloc[:, 1:])

PCA(copy=True, n_components=None, whiten=False)

# データを主成分空間に写像 = 次元圧縮
feature = pca.transform(df.iloc[:, 1:])

# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
for x, y, name in zip(feature[:, 0], feature[:, 1], df.iloc[:, 0]):
    plt.text(x, y, name, alpha=0.8, size=10)
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, color=colors)
plt.title("Principal Component Analysis")
plt.xlabel("The first principal component score")
plt.ylabel("The second principal component score")
plt.savefig('k-means/PCA12_plot.jpg')
plt.pause(1)
plt.close()

