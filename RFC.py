from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split, GridSearchCV

mnist = datasets.load_digits()
 
train_images, test_images, train_labels, test_labels = \
train_test_split(mnist.data, mnist.target, test_size=0.2)

plt.figure(figsize=(15,15))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i].reshape((8,8)), cmap=plt.cm.binary)
    plt.xlabel(train_labels[i], fontsize=18)
plt.savefig('RFC/dataset_input.jpg')
plt.pause(1)
plt.close()

clf = RFC(verbose=True,       # 学習中にログを表示します。この指定はなくてもOK
          n_jobs=-1,          # 複数のCPUコアを使って並列に学習します。-1は最大値。
          random_state=2525)  # 乱数のシードです。
clf.fit(train_images, train_labels)

print(clf.feature_importances_)


#print(f"acc: {clf.score(test_images, test_labels)}")
acc=clf.score(test_images, test_labels)*100
print("acc:{:.2f} %".format(acc))

predicted_labels = clf.predict(test_images)

plt.figure(figsize=(15,15))
# 先頭から25枚テストデータを可視化
for i in range(25):
 
    # 画像を作成
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i].reshape((8,8)), cmap=plt.cm.binary)
 
    # 今プロットを作っている画像データの予測ラベルと正解ラベルをセット
    predicted_label = predicted_labels[i]
    true_label      = test_labels[i]
 
    # 予測ラベルが正解なら緑、不正解なら赤色を使う
    if predicted_label == true_label:
        color = 'green' # True label color
    else:
        color = 'blue'   # False label color red
    plt.xlabel("{} True({})".format(predicted_label,
                                  true_label), color=color, fontsize=18)
    if i==0:
        plt.title("acc:{:.2f} %".format(acc), fontsize=36)  
 
plt.savefig('RFC/RFC_results.jpg')
plt.pause(1)
plt.close()

search_params = {
     'n_estimators'      : [75,100,150],   
      'max_features'      : [8,10,12],  
      'random_state'      : [2525],
      'n_jobs'            : [1],
      'min_samples_split' : [3,4,5],  
      'max_depth'         : [13,15,18]  
}
"""
search_params = {
     'n_estimators'      : [5, 10, 20, 30, 50, 100, 300],   
      'max_features'      : [3, 5, 10, 15, 20],  
      'random_state'      : [2525],
      'n_jobs'            : [1],
      'min_samples_split' : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100],  
      'max_depth'         : [3, 5, 10, 15, 20, 25, 30, 40, 50, 100]  
}
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=15, max_features=10, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=3,
            min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
            oob_score=False, random_state=2525, verbose=0,
            warm_start=False)
acc:97.22 %
"""

gs = GridSearchCV(RFC(),           # 対象の機械学習モデル
                  search_params,   # 探索パラメタ辞書
                  cv=3,            # クロスバリデーションの分割数 3
                  verbose=True,    # ログ表示verbose=True
                  n_jobs=1)       # 並列処理 最大値-1
gs.fit(train_images, train_labels)
 
print(gs.best_estimator_)

acc=gs.score(test_images, test_labels)*100
print("acc:{:.2f} %".format(acc))

predicted_labels = gs.predict(test_images)

plt.figure(figsize=(15,15))
# 先頭から25枚テストデータを可視化
for i in range(25):
 
    # 画像を作成
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(test_images[i].reshape((8,8)), cmap=plt.cm.binary)
 
    # 今プロットを作っている画像データの予測ラベルと正解ラベルをセット
    predicted_label = predicted_labels[i]
    true_label      = test_labels[i]
 
    # 予測ラベルが正解なら緑、不正解なら赤色を使う
    if predicted_label == true_label:
        color = 'green' # True label color
    else:
        color = 'blue'   # False label color red
    plt.xlabel("{} True({})".format(predicted_label,
                                  true_label), color=color, fontsize=18)
    if i==0:
        plt.title("acc:{:.2f} %".format(acc), fontsize=36)  
 
plt.savefig('RFC/RFC_best_results.jpg')
plt.pause(1)
plt.close()
 