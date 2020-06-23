#import 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

# SVMクラス
#    kernel: カーネル
#    gamma: 境界の複雑さ
#    degree: カーネル多項式の度合
class SVM():
    def __init__(self, c=1, kernel='rbf', gamma=0.1, degree=3):
        print(f"kernel : {kernel}")
        self.clf = svm.SVC(C=c,
                           kernel=kernel,
                           gamma=gamma, 
                           degree=degree)

    # 予測
    def __call__(self, test_data):
        predict_data = self.clf.predict(test_data)
        return predict_data

    # 学習
    def training(self, train_data, train_target):
        self.clf.fit(train_data, train_target)

def main():
    # -------------------------------------
    # init setting
    # -------------------------------------

    # パラメータ
    parser = argparse.ArgumentParser(description = 
                                     "SVM using iris_dataset as an exam")
    parser.add_argument("-c", "--c",
                        help = "param C", 
                        type = float, 
                        default = 1)
    parser.add_argument("-k", "--kernel", 
                        help = "param kernel(linear, poly, rbf or sigmoid)", 
                        type = str, 
                        default = 'rbf')
    parser.add_argument("-g", "--gamma", 
                        help = "param gamma, not need linear", 
                        type = float, 
                        default = 0.1)
    parser.add_argument("-d", "--degree", 
                        help = "param degree, only use poly", 
                        type = int, 
                        default = 3)
    args = parser.parse_args()

    kernel = args.kernel
    c = args.c 
    gamma = args.gamma
    degree = args.degree 

    # -------------------------------------
    # dataset
    # -------------------------------------
    # irisデータセットを読み込む
    iris = datasets.load_iris()
    # データは特徴量を2つに絞る
    data = iris.data[:, 0:2]
    # 教師データ
    target = iris.target

    # -------------------------------------
    # train
    # -------------------------------------
    # SVMクラスのインスタンスを作成
    clf = SVM(c, kernel, gamma, degree)
    # SVMの学習
    clf.training(data, target)

    # plot用にdataを分割
    x = data[:, 0]
    y = data[:, 1]
    # meshを作成
    x_min, x_max = x.min() - .5, x.max() + .5
    y_min, y_max = y.min() - .5, y.max() + .5
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                 np.arange(y_min, y_max, 0.05))

    # 学習済みモデルによる予測
    z = clf(np.c_[x_mesh.ravel(), y_mesh.ravel()])
    z = z.reshape(x_mesh.shape)

    # -------------------------------------
    # plot
    # -------------------------------------
    
    # plotの時に色づけするためのカラーマップ
    color = ['k', 'g', 'r']
    color_list = [color[i] for i in target]

    # figureを作成
    plt.figure()
    # 散布図を描画
    plt.scatter(x, y, color = color_list)
    # 境界線の描画
    plt.contourf(x_mesh, y_mesh, z, cmap=plt.cm.bone, alpha=0.4)
    # titleを表示
    plt.title(f'{kernel}-SVC')
    # 表示
    plt.show()

if __name__=='__main__':
    main()
