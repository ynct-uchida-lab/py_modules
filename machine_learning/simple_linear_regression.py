#import
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston

class SimpleLinearRegression():
    def __init__(self, dataframe, X_feature, test_size, y_feature):
        #データを分割
        train, test = train_test_split(dataframe, 
                                       test_size = test_size, 
                                       random_state = 0)

        self.X_train = train[X_feature].values.reshape(-1,1)
        self.y_train = train[y_feature]
        self.X_test = test[X_feature].values.reshape(-1,1)
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


def main() :
    #boston住宅価格データセットを例に単回帰分析する

    #パラメータ
    parser = argparse.ArgumentParser(description= 
                                     "simple_reg using boston_dataset as an exam")
    parser.add_argument("-x", "--X_feature",
                        help = "feature for predition(df = RM)",
                        type = str, 
                        default = "RM")
    parser.add_argument("-s", "--test_size",
                        help = "test size(df = 0.3)",
                        type = float,
                        default = 0.3)
    args = parser.parse_args()

    X_feature = args.X_feature #予測するための特徴量
    test_size = args.test_size #テストデータの大きさ


    boston = load_boston()

    #データフレーム
    boston_dataframe = pd.DataFrame(boston.data, columns = boston.feature_names)
    boston_dataframe['PRICE'] = boston.target

    #単回帰分析->メソッド
    model = SimpleLinearRegression(boston_dataframe, X_feature, test_size, y_feature = 'PRICE')
    model.training()

    #scoreで評価->メソッド
    model.score()

if __name__ == '__main__':
    main()