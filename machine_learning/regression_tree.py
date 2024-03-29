# import
import argparse  # コマンドライン引数用
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import tree
from sklearn.tree import plot_tree  # 可視化用


# 決定木分析 : Regression Tree クラス
class RegressionTree():
    def __init__(self, max_depth):
        self.max_depth = max_depth

        # モデルの作成
        self.clf = tree.DecisionTreeRegressor(max_depth=self.max_depth)

    # 予測
    def __call__(self, data):
        predict = self.clf.predict(data)
        return predict

    # 学習
    def training(self, train_data, train_target):
        self.clf.fit(train_data, train_target)

    # 可視化
    def visualizetree(self, class_names, feature_names):
        fig_tree = plt.figure(figsize=(16, 8))
        ax_tree = fig_tree.add_subplot(111)
        plot_tree(self.clf,
                  class_names=class_names,
                  feature_names=feature_names,
                  filled=True,
                  rounded=True,
                  proportion=True,
                  fontsize=10.5,
                  ax=ax_tree
                  )
        plt.show()


def main():
    # パラメータ
    parser = argparse.ArgumentParser(
        description="RegressionTree using iris_dataset as an exam")
    parser.add_argument("-d", "--max_depth",
                        help="tree depth (default = 3)",
                        type=int,
                        default=3)
    parser.add_argument("-nv", "--visualization",
                        help="T/F show tree. (T:defo, F:'-nv')",
                        action="store_false")
    args = parser.parse_args()

    max_depth = args.max_depth  # 決定木の深さ
    visualize = args.visualization  # 可視化の可否

    # irisデータセットを例に決定木分析する
    iris = datasets.load_iris()

    data = iris.data[:, [0, 2]]
    target = iris.target

    # 決定木分析
    clf = RegressionTree(max_depth)  # -> __init__
    clf.training(data, target)  # -> training

    # plot用にdataを分割
    x = data[:, 0]
    y = data[:, 1]

    # meshを作成
    x_min, x_max = x.min() - .5, x.max() + .5
    y_min, y_max = y.min() - .5, y.max() + .5
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                 np.arange(y_min, y_max, 0.05))

    # __call__
    z = clf(np.c_[x_mesh.ravel(), y_mesh.ravel()])
    z = z.reshape(x_mesh.shape)

    # plotの時に色づけするためのカラーマップ
    color = ['k', 'g', 'r']
    color_list = [color[i] for i in target]

    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, color=color_list)

    # 境界線の描画
    plt.contourf(x_mesh, y_mesh, z, cmap=plt.cm.Paired, alpha=0.4)
    # titleを表示
    plt.title(f'max_depth = {max_depth}')

    plt.show()

    # 可視化
    if (visualize):
        target_names = iris.target_names
        feature_names = iris.feature_names
        clf.visualizetree(target_names, feature_names)


if __name__ == '__main__':
    main()
