import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn import datasets
from sklearn.cross_validation import train_test_split # クロスバリデーション用
from sklearn.svm import SVC # SVM用
from sklearn import metrics       # 精度検証用
import pandas as pd
import seaborn as sns

# データ用意
iris = datasets.load_iris()    # データロード
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target_names[iris.target]
#df.head()
print(df.head())

X = iris.data                  # 説明変数セット
Y = iris.target                # 目的変数セット
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0) # random_stateはseed値。
print(len(X_train),len(Y_train),len(X_test),len(Y_test))

# SVM実行
model = SVC()               # インスタンス生成
model.fit(X_train, Y_train) # SVM実行

# 予測実行
predicted = model.predict(X_test) # テストデータでの予測実行
acc=metrics.accuracy_score(Y_test, predicted)
print("acc=",acc)

from sklearn.neighbors import KNeighborsClassifier
#from sklearn.cross_validation import train_test_split # trainとtest分割用

# train用とtest用のデータ用意。test_sizeでテスト用データの割合を指定。random_stateはseed値を適当にセット。
#X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state=3) 

knn = KNeighborsClassifier(n_neighbors=6) # インスタンス生成。n_neighbors:Kの数
knn.fit(X_train, Y_train)                 # モデル作成実行
Y_pred = knn.predict(X_test)              # 予測実行

# 精度確認用のライブラリインポートと実行
#from sklearn import metrics
acc1=metrics.accuracy_score(Y_test, Y_pred)    # 予測精度計測
print("acc1=",acc1)