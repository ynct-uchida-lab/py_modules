#import 
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

#SVMクラス
class SVM():
    def __init__(self, C = 1, kernel = 'rbf', gamma = 0.1, degree = 3):
        self._C = C
        self._kernel = kernel
        self._gamma = gamma
        self._degree = degree

        print(f"kernel : {self._kernel}")

        #学習器の作成
        if(self._kernel == 'linear'):
            self.clf = svm.SVC(C = self._C,
                               kernel = self._kernel)
        elif(self._kernel == 'poly'):
            self.clf = svm.SVC(C = self._C,
                               kernel = self._kernel,
                               gamma = self._gamma, 
                               degree = self._degree)
        elif(self._kernel == 'sigmoid'):
            self.clf = svm.SVC(C = self._C,
                               kernel = self._kernel,
                               gamma = self._gamma)
        else: #rbf
            self.clf = svm.SVC(C = self._C,
                               kernel = self._kernel,
                               gamma = self._gamma)

    #予測
    def __call__(self, test_data):
        predict_data = self.clf.predict(test_data)
        return predict_data

    #学習
    def training(self, train_data, train_target):
        self.clf.fit(train_data, train_target)


def main():
    #パラメータ
    parser = argparse.ArgumentParser(description = 
                                     "SVM using iris_dataset as an exam")

    parser.add_argument("-c", "--C",
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
    C = args.C 
    gamma = args.gamma
    degree = args.degree 

    
    iris = datasets.load_iris()

    data = iris.data[:, 0:2] #特徴量を2つに絞る
    target = iris.target

    clf = SVM(C, kernel, gamma, degree)
    clf.training(data, target)

    #plot用にdataを分割
    x = data[:, 0]
    y = data[:, 1]
    
    #meshを作成
    x_min, x_max = x.min() - .5, x.max() + .5
    y_min, y_max = y.min() - .5, y.max() + .5
    x_mesh, y_mesh = np.meshgrid(np.arange(x_min, x_max, 0.05),
                                 np.arange(y_min, y_max, 0.05))
                            
    #__call__
    z = clf(np.c_[x_mesh.ravel(), y_mesh.ravel()])
    z = z.reshape(x_mesh.shape)
    
    #plotの時に色づけするためのカラーマップ
    color = ['k', 'g', 'r']
    color_list = [color[i] for i in target]

    plt.figure(figsize = (4, 4))
    plt.scatter(x, y, color = color_list)

    #境界線の描画
    plt.contourf(x_mesh, y_mesh, z, cmap=plt.cm.bone, alpha = 0.4)
    #titleを表示
    plt.title(f'{kernel}-SVC')

    plt.show()

if __name__=='__main__':
    main()