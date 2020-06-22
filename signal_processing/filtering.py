import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

#データに指定されたフィルタをかける関数
def butterworth(x, samplerate, fp, fs, gpass, gstop, btype):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, btype)            #フィルタ伝達関数の分子と分母を計算
    data_lpf = signal.filtfilt(b, a, x)           #信号に対してフィルタをかける
    return data_lpf                               #フィルタ後の信号を返す

def main():
    samplerate = 25600                                   #波形のサンプリングレート
    x = np.arange(0, 25600) / samplerate                 #波形生成のための時間軸の作成
    data = np.random.normal(loc=0, scale=1, size=len(x)) #ガウシアンノイズを生成

    dt=1/samplerate                               #サンプリング間隔
    t=np.arange(0,samplerate*dt,dt)               #時間軸
 
    fp = 3000                                     #通過域端周波数[Hz]
    fs = 6000                                     #阻止域端周波数[Hz]
    gpass = 3                                     #通過域端最大損失[dB]
    gstop = 40                                    #阻止域端最小損失[dB]
    
    #データにローパスフィルタをかける
    data_lpf = butterworth(data, samplerate, fp, fs, gpass, gstop, 'low')
    #データにハイパスフィルタをかける
    data_hpf = butterworth(data, samplerate, fp, fs, gpass, gstop, 'high')
    
    plt.subplots_adjust(hspace=0.6)
    
    plt.subplot(2,2,1)
    plt.plot(t,data)
    plt.title("original",y=-0.4)
    
    plt.subplot(2,2,3)
    plt.plot(t,data_lpf,color='cyan')
    plt.title("after lpf",y=-0.4)
    
    plt.subplot(2,2,4)
    plt.plot(t,data_hpf,color='red')
    plt.title("after hpf",y=-0.4)
    
    plt.show()
    
if  __name__  ==  "__main__":
    main()