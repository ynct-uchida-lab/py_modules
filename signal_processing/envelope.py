import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# エンベロープ波形を求める関数
def envelope(data):
    # データをヒルベルト変換する
    data_hlb = signal.hilbert(data)
    # ヒルベルト変換したデータの絶対値をとりエンベロープ波形を求める
    data_evl = np.abs(data_hlb)
    return data_evl

def sample_code():
    # -------------------------------------
    # サンプルデータの生成
    # -------------------------------------
    # サンプリング周波数
    dt = 0.001
    # 時間の配列
    time_array = np.arange(0, 1, dt)

    # データの生成
    data = np.cos(2 * np.pi * 50 * time_array)
    data *= (1 + 0.5 * np.sin(2 * np.pi * 2 * time_array))
    data *= (1 + 0.5 * np.sin(2 * np.pi * 10 * time_array))
    
    # サンプルデータのエンベロープ波形を求める
    data_evl = envelope(data)
    
    # -------------------------------------
    # 描画
    # -------------------------------------
    plt.plot(time_array, data, c='k', label='Original wave')
    plt.plot(time_array, data_evl, c='r', label='Envelope wave')
    plt.ylabel('Amplitude')
    plt.xlabel('Time [s]')
    plt.xlim([0.0, 1.0])
    plt.ylim([-2.5, 2.5])
    plt.legend()
    plt.show()

def main():
    sample_code()

if __name__ == "__main__":
    main()
