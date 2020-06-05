# import 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import svm
from sklearn import datasets

# One Class SVMクラス
#    nu: 異常値の割合(0 - 1)
#    kernel: カーネル
#    gamma: 境界の複雑さ
class OneClassSVM():
    def __init__(self, nu=0.1, kernel='rbf', gamma=0.1):
        self._nu = nu
        self._kernel = kernel
        self._gamma = gamma

        # 学習器の作成
        self.clf = svm.OneClassSVM(
                        nu=self._nu,
                        kernel=self._kernel,
                        gamma=self._gamma
                    )

    # 予測
    def __call__(self, data):
        predict_data = self.clf.predict(data)
        return predict_data

    # 学習
    def training(self, train_data):
        self.clf.fit(train_data)

# main
def main():
    
    # irisデータセットを読み込む
    iris = datasets.load_iris()
    # 2列目までをデータとして格納する（特徴量2にする）
    data = iris.data[:, 0:2]
    # 教師データを格納する
    target = iris.target
    
    # OneClassSVMのインスタンスを生成
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.5)
    # OneClassSVMの学習(データを使う)
    clf.training(data[:50, :])

    # plotの時に色づけするためのカラーマップ
    #     0: 黒, 1: 緑, 2: 赤
    color = ['k', 'g', 'r']
    color_list = [color[i] for i in target]
    
    # plot用にdataを分割
    x = data[:, 0]
    y = data[:, 1]

    # 散布図をplot
    plt.scatter(x, y, color=color_list)

    # meshを作成する
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                 np.arange(y_min, y_max, 0.05))
    
    # 境界線のplot
    z = clf(np.c_[x_mesh.ravel(), y_mesh.ravel()])
    z = z.reshape(x_mesh.shape)
    plt.contourf(x_mesh, y_mesh, z, cmap=plt.cm.Paired, alpha=0.3)

    # 表示
    plt.show()

if __name__=='__main__':
    main()

