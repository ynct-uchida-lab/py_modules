import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from fft import fft              #fft.pyのfft関数を呼び出す

#データにローパスフィルタ(バターワースフィルタ）をかける関数
def lpf(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "band")           #フィルタ伝達関数の分子と分母を計算
    data_lpf = signal.filtfilt(b, a, x)           #信号に対してフィルタをかける
    return data_lpf                               #フィルタ後の信号を返す

def main():
    samplerate = 25600                                   #波形のサンプリングレート
    x = np.arange(0, 25600) / samplerate                 #波形生成のための時間軸の作成
    data = np.random.normal(loc=0, scale=1, size=len(x)) #ガウシアンノイズを生成

    dt=1/samplerate                               #サンプリング間隔
    t=np.arange(0,samplerate*dt,dt)               #時間軸
 
    fp = np.array([1000,3000])     #通過域端周波数[Hz]
    fs = np.array([500,600])       #阻止域端周波数[Hz]
    gpass = 1                      #通過域端最大損失[dB]
    gstop = 40                     #阻止域端最小損失[dB]
    
    #データにローパスフィルタをかける
    data_lpf = lpf(data, samplerate, fp, fs, gpass, gstop)
    #データを高速フーリエ変換する
    data_fft = fft(data)
    #ローパスフィルタがかけられたデータを高速フーリエ変換する
    data_lpf_fft = fft(data_lpf)
    
    plt.subplots_adjust(hspace=1.0)
    
    plt.subplot(3,1,1)
    plt.plot(t,data,label="original")
    plt.plot(t,data_lpf,label="filter")
    plt.title("data",y=-0.7)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=8)
    
    plt.subplot(3,1,2)
    plt.plot(data_fft,label="original")
    plt.plot(data_lpf_fft,label="filter")
    plt.title("spectrum",y=-0.7)
    plt.xlim(0,12800)
    
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0, fontsize=8)
    
    plt.show()
    
if  __name__  ==  "__main__":
    main()