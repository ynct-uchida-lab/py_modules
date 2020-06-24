import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# エンベロープ波形を求める関数
def envelope(data):
    # データをヒルベルト変換する
    data_hlb = signal.hilbert(data)
    #ヒルベルト変換したデータの絶対値をとり、エンベロープ波形を求める
    data_evl = np.abs(data_hlb)
    return data_evl

def main():
    # サンプルデータの生成
    dt = 0.001                             # サンプリング周波数
    t = np.arange(0, 1, dt)                # 時間軸
    data = np.cos(2 * np.pi * 50 * t)
    data *= (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    data *= (1 + 0.5 * np.sin(2 * np.pi * 10 * t))
    
    # サンプルデータのエンベロープ波形を求める
    data_evl = envelope(data)
    
    plt.plot(t, data, label='original')
    plt.plot(t, data_evl, label='envelope')
    plt.ylabel("y")
    plt.xlabel("time[s]")

    plt.legend()
    plt.show()

if  __name__  ==  "__main__":
    main()