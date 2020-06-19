﻿import numpy as np
import matplotlib.pyplot as plt

# データを高速フーリエ変換しスペクトルを求める関数
def fft(data):
    data_fft=np.fft.fft(data)
    # 振幅を求めるために高速フーリエ変換されたデータを正規化
    spectrum = np.abs(data_fft)
    spectrum = spectrum[:int(len(spectrum)/2)]
    return spectrum
    
def main():
    # サンプルデータの作成
    dt = 0.01 # サンプリング周波数
    tm = 1 # データの収録時間
    f1, f2 = 5, 8 # 周波数
    a1, a2 = 5, 4 # 信号の振幅

    t = np.arange(0, tm, dt) # 時間
    data = a1*np.sin(2*np.pi*f1*t) + a2*np.sin(2*np.pi*f2*t) # サンプルデータ
    freq = np.fft.fftfreq(len(data), dt) # 周波数
    freq = freq[:int (len(freq)/2)]
    
    #サンプルデータを高速フーリエ変換（FFT)
    spectrum = fft(data)

    #グラフをプロットする
    plt.figure(2)
    plt.subplot(211)
    plt.plot(t,data)
    plt.xlabel("time")
    plt.ylabel("amplitude")

    plt.subplot(212)
    plt.plot(freq,spectrum)
    plt.xlabel("frequency")
    plt.ylabel("amplitude")

    plt.tight_layout()
    
    plt.show()

if  __name__  ==  "__main__":
    main()