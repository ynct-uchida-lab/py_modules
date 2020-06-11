#import
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

#Multiple Linear Regression クラス
class MultipleLinearRegression():
    def __init__(self, dataframe, X_features, test_size, y_feature):
        #データを分割
        train, test = train_test_split(dataframe, 
                                       test_size = test_size, 
                                       random_state = 0)

        self.X_train = train[X_features]
        self.y_train = train[y_feature]
        self.X_test = test[X_features]
        self.y_test = test[y_feature]

        #線形回帰モデル
        self.model = LinearRegression()
    
    #学習
    def training(self):
        self.model.fit(self.X_train, self.y_train)

    #scoreで評価
    def score(self):
        print(f'訓練用データ : {self.model.score(self.X_train, self.y_train)}')
        print(f'テスト用データ : {self.model.score(self.X_test, self.y_test)}')


def main():
    #boston住宅価格データセットを例に重回帰分析する

    #パラメータ
    parser = argparse.ArgumentParser(description= 
                                     "multiple_reg using boston_dataset as an exam")
    parser.add_argument("-x", "--X_features",
                        help = "feature for predition(df = 13 features)\
                                \n※how2discribe: -x hoge -x fuga",
                        action = 'append', 
                        default = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
    parser.add_argument("-s", "--test_size",
                        help = "test size(df = 0.3)",
                        type = float,
                        default = 0.3)
    args = parser.parse_args()

    X_features = args.X_features #予測するための特徴量
    test_size = args.test_size #テストデータの大きさ

    #コマンドライン引数が指定されるとdefaultのリストの後ろに要素が追加されるため対処
    if (len(X_features) >= 14):
        del X_features[:13] #先頭から12個の特徴量を削除(-> default値を消す)


    boston = load_boston()

    #データフレーム
    boston_dataframe = pd.DataFrame(boston.data, columns = boston.feature_names)
    boston_dataframe['PRICE'] = boston.target #PRICE(住宅価格)を追加

    #重回帰分析->メソッド
    model = MultipleLinearRegression(boston_dataframe, X_features, test_size, y_feature = 'PRICE')
    model.training()

    #scoreで評価->メソッド
    model.score()

if __name__ == '__main__':
    main()