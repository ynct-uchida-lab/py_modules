# import
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets
import random


# linear Regression クラス
class RegressionAnalysis():
    def __init__(self, dataframe, X_feature, y_feature='PRICE', test_size=0.3):
        # データを分割
        train, test = train_test_split(dataframe,
                                       test_size=test_size,
                                       random_state=0)

        # route simple linear reg
        if type(X_feature) is str:
            self.X_train = train[X_feature].values.reshape(-1, 1)
            self.y_train = train[y_feature]
            self.X_test = test[X_feature].values.reshape(-1, 1)
            self.y_test = test[y_feature]

        # route multiple linear reg
        else:
            self.X_train = train[X_feature]
            self.y_train = train[y_feature]
            self.X_test = test[X_feature]
            self.y_test = test[y_feature]

        # 線形回帰モデル
        self.model = LinearRegression()

    # 予測
    def __call__(self, test_data):
        predict_data = self.model.predict(test_data)
        return predict_data

    # 学習
    def training(self):
        self.model.fit(self.X_train, self.y_train)

    # scoreで評価
    def score(self):
        train_score = self.model.score(self.X_train, self.y_train)
        test_score = self.model.score(self.X_test, self.y_test)

        return train_score, test_score


def main():
    # boston住宅価格データセットを例に回帰分析する

    # パラメータ
    parser = argparse.ArgumentParser(
        description="linear_reg using boston_dataset as an exam")

    parser.add_argument("-mx", "--X_features",
                        help="feature for predition(df = 13 features)\
                                \n※how2discrib: -mx hoge -mx fuga\
                                \nChoose from CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT",
                        action='append',
                        default=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'])
    parser.add_argument("-x", "--X_feature",
                        help="feature for predition(df = RM)",
                        type=str,
                        default="RM")
    parser.add_argument("-ts", "--test_size",
                        help="test size(df = 0.3)",
                        type=float,
                        default=0.3)
    args = parser.parse_args()

    X_features = args.X_features  # 予測するための特徴量
    X_feature = args.X_feature  # 予測するための特徴量
    test_size = args.test_size  # テストデータの大きさ

    # コマンドライン引数が指定されるとdefaultのリストの後ろに要素が追加されるため対処
    if (len(X_features) >= 14):
        del X_features[:13]  # 先頭から12個の特徴量を削除(-> default値を消す)

    boston = datasets.load_boston()

    # データフレーム
    boston_dataframe = pd.DataFrame(boston.data, columns=boston.feature_names)
    boston_dataframe['PRICE'] = boston.target  # PRICE(住宅価格)を追加

    # 目的変数にPRICEを指定
    y_feature = 'PRICE'

    # テストデータをサンプルとしてランダムに抽出するための乱数
    rand = random.randrange(len(boston_dataframe))

    for i in range(2):
        # 単回帰分析->メソッド
        if i == 0:
            print('単回帰分析-----------------------')
            # 予測用test_data
            test_data = [boston_dataframe.loc[rand, X_feature]]

        # 重回帰分析->メソッド
        else:
            print('\n重回帰分析-----------------------')
            X_feature = X_features

            # 予測用test_dataリストを作成
            for x in range(len(X_feature)):
                each_data = boston_dataframe.loc[rand, X_feature[x]]
                test_data.append(each_data)

        model = RegressionAnalysis(boston_dataframe,
                                   X_feature,
                                   y_feature='PRICE',
                                   test_size=test_size)

        # 学習
        model.training()

        # 予測
        pred = model([test_data])

        # scoreで評価->メソッド, 予測結果の表示
        train_score, test_score = model.score()
        print(f"学習スコア   : {train_score}\
                \nテストスコア : {test_score}\
                \n\ntest_data    : {test_data}\
                \n予測結果     : {pred} <---> correct : {boston_dataframe.loc[rand, y_feature]}")  # 正解データと比較

        # 予測用test_dataの要素を全消去(単回帰で用いたtest_dataを捨てる)
        test_data.clear()


if __name__ == '__main__':
    main()
