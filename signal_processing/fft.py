import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# データを高速フーリエ変換しスペクトルを求める関数
def fft(data):
    data_fft = np.fft.fft(data)
    # 振幅を求めるために高速フーリエ変換されたデータを正規化
    spectrum = np.abs(data_fft)
    spectrum = spectrum[:int(len(spectrum) / 2)]
    return spectrum

# データのパワースペクトルを求める関数
def powerspectrum(data, fs, type):
    freq, P = signal.periodogram(data, fs)
    
    # typeによってパワーを変更する
    if (type == "dB"):
        return freq, 10*np.log10(P)
    else:
        return freq, P
def main():
    # サンプルデータの作成
    dt = 0.01      # サンプリング周期
    fs = 1 / dt    # サンプリング周波数
    tm = 1         # データの収録時間
    f1, f2 = 5, 8  # 周波数
    a1, a2 = 5, 4  # 信号の振幅

    t = np.arange(0, tm, dt)                                                  # 時間
    data = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)  # サンプルデータ
    freq = np.fft.fftfreq(len(data), dt)                                      # 周波数
    freq = freq[:int(len(freq) / 2)]
    
    # サンプルデータを高速フーリエ変換（FFT)
    spectrum = fft(data)
    
    # パワースペクトルを求める
    freq_P, P = powerspectrum(data, fs, "none")
    freq_P_dB, P_dB = powerspectrum(data, fs, "dB")

    # グラフをプロットする
    plt.subplots_adjust(wspace=0.4, hspace=1.0)

    plt.subplot(321)
    plt.plot(t, data)
    plt.xlabel("time")
    plt.ylabel("amplitude")
    plt.title("original")

    plt.subplot(323)
    plt.plot(freq, spectrum)
    plt.xlabel("frequency")
    plt.ylabel("amplitude")
    plt.title("spectrum")
    
    plt.subplot(325)
    plt.plot(freq_P, P)
    plt.xlabel("Frequency")
    plt.ylabel("Power/frequency")
    plt.title("power spectrum")

    plt.subplot(326)
    plt.plot(freq_P_dB, P_dB)
    plt.xlabel("Frequency")
    plt.ylabel("Power/frequency")
    plt.title("power spectrum [dB]")
    
    plt.show()

if __name__ == "__main__":
    main()
