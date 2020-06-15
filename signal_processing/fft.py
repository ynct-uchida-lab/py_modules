import numpy as np
import matplotlib.pyplot as plt

def fft(data):
    data_fft=np.fft.fft(data)
    # 振幅を求めるためにfftされたデータを正規化
    spectrum = np.abs(data_fft)
    return spectrum
    
def main():
    # サンプルデータの作成
    dt = 0.01 # サンプリング周波数
    tm = 1 # データの収録時間
    N = tm / dt # サンプル数
    f1, f2 = 5, 8 # 周波数
    A1, A2 = 5, 4 # 振幅

    t = np.arange(0, tm, dt) # 時間
    data = A1*np.sin(2*np.pi*f1*t) + A2*np.sin(2*np.pi*f2*t) # サンプルデータ
    freq = np.fft.fftfreq(len(data), dt) # 周波数
    
    #サンプルデータを高速フーリエ変換（FFT)
    spectrum=fft(data)

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
    plt.savefig("01")
    
    plt.show()

if  __name__  ==  "__main__":
    main()