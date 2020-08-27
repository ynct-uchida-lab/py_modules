import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# データを高速フーリエ変換しスペクトルを求める関数
def fft(data, dtype=None):
    data_fft = np.fft.fft(data)
    # 振幅を求めるために高速フーリエ変換されたデータを正規化
    spectrum = np.abs(data_fft)
    spectrum = spectrum[:int(len(spectrum) / 2)]

    return spectrum

# 単位を'dB'に変換
def amp_to_db(spectrum):
    return 20 * np.log10(spectrum)

def main():
    # サンプルデータの作成
    dt = 0.01      # サンプリング周期
    fs = 1 / dt    # サンプリング周波数
    tm = 1         # データの収録時間
    f1, f2 = 5, 8  # 周波数
    a1, a2 = 5, 4  # 信号の振幅

    # 時間
    t = np.arange(0, tm, dt)
    # サンプルデータ
    data = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    # 周波数
    freq = np.fft.fftfreq(len(data), dt)
    freq = freq[:int(len(freq) / 2)]
    
    # サンプルデータを高速フーリエ変換（FFT)
    spectrum = fft(data)
    
    # dB表記に直す
    power = amp_to_db(spectrum)

    # グラフをプロットする
    plt.subplots_adjust(wspace=0.4, hspace=1.0)

    plt.subplot(3, 1, 1)
    plt.plot(t, data)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("original")

    plt.subplot(3, 1, 2)
    plt.plot(freq, spectrum)
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("spectrum")
    
    plt.subplot(3, 1, 3)
    plt.plot(freq, power)
    plt.xlabel("Frequency")
    plt.ylabel("dB")
    plt.title("power spectrum")

    plt.show()

if __name__ == "__main__":
    main()
