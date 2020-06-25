#import
import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import datasets

#Linear Regression クラス
class SampleLinearRegression():
    def __init__(self, dataframe, X_feature, y_feature, test_size=0.3, multiple_mode=False):
        #データを分割
        train, test = train_test_split(dataframe, 
                                       test_size = test_size, 
                                       random_state = 0)

        if multiple_mode is False:
            self.X_train = train[X_feature].values.reshape(-1,1)
            self.y_train = train[y_feature]
            self.X_test = test[X_feature].values.reshape(-1,1)
            self.y_test = test[y_feature]
        
        else:
            self.X_train = train[X_feature]
            self.y_train = train[y_feature]
            self.X_test = test[X_feature]
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
    #boston住宅価格データセットを例に回帰分析する

    #パラメータ
    parser = argparse.ArgumentParser(description= 
                                     "linear_reg using boston_dataset as an exam")
    
    parser.add_argument("-mx", "--X_features",
                        help = "feature for predition(df = 13 features)\
                                \n※how2discribe: -mx hoge -mx fuga",
                        action = 'append', 
                        default = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
    parser.add_argument("-x", "--X_feature",
                        help = "feature for predition(df = RM)",
                        type = str, 
                        default = "RM")
    parser.add_argument("-ts", "--test_size",
                        help = "test size(df = 0.3)",
                        type = float,
                        default = 0.3)
    args = parser.parse_args()

    X_features = args.X_features #予測するための特徴量
    X_feature = args.X_feature #予測するための特徴量
    test_size = args.test_size #テストデータの大きさ

    #コマンドライン引数が指定されるとdefaultのリストの後ろに要素が追加されるため対処
    if (len(X_features) >= 14):
        del X_features[:13] #先頭から12個の特徴量を削除(-> default値を消す)

    boston = datasets.load_boston()

    #データフレーム
    boston_dataframe = pd.DataFrame(boston.data, columns = boston.feature_names)
    boston_dataframe['PRICE'] = boston.target #PRICE(住宅価格)を追加
    
    for i in range(2):
        if i == 0:
            #単回帰分析->メソッド
            print('単回帰分析')
            model = SampleLinearRegression(boston_dataframe, X_feature, y_feature = 'PRICE', test_size = test_size)
            model.training()

        else:
            #重回帰分析->メソッド
            print('重回帰分析')
            model = SampleLinearRegression(boston_dataframe, X_features, y_feature = 'PRICE', test_size = test_size, multiple_mode = True)
            model.training()

        #scoreで評価->メソッド
        model.score()

if __name__=='__main__':
    main()