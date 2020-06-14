import numpy as np
import matplotlib.pyplot as plt

def fft(data):
    F=np.fft.fft(data)
    
    # 振幅を求めるためにfftされたデータを正規化
    F_abs = np.abs(F)
    N=2**20 #サンプル数
    F_abs_amp = F_abs / N*2
    F_abs_amp[0] = F_abs_amp[0] / 2
    return F_abs_amp
    
def main():
    # サンプルデータの作成
    N = 2**20 # サンプル数
    dt = 0.0001 # サンプリング周波数
    f1, f2 = 5, 8 # 周波数
    A1, A2 = 5, 0 # 振幅
    p1, p2 = 0, 0 # 位相

    t = np.arange(0, N*dt, dt) # time
    freq = np.linspace(0, 1.0/dt, N) # frequency step

    data = A1*np.sin(2*np.pi*f1*t + p1) + A2*np.sin(2*np.pi*f2*t + p2) 
    
    #サンプルデータを高速フーリエ変換（FFT)
    data_f=fft(data)
    
    #グラフをプロットする
    plt.figure(2)
    plt.subplot(211)
    plt.plot(t,data)
    plt.xlim(0, 1)
    plt.xlabel("time")
    plt.ylabel("amplitude")

    plt.subplot(212)
    plt.plot(freq,data_f)
    plt.xlim(0, 10)
    plt.xlabel("frequency")
    plt.ylabel("amplitude")

    plt.tight_layout()
    plt.savefig("01")
    
    plt.show()

if  __name__  ==  "__main__":
    main()