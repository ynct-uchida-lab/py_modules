# import
import numpy as np
import matplotlib.pyplot as plt

# Adaptive Filter: NLMS algorithm
# inputs
#   d: Desired response
#   x: Input data
#   n: Filter length
#   mu: step size
# output
#   s: output data(signal)
def nlms(d, x, n, mu):
    
    # -------------------------------------
    # 初期化
    # -------------------------------------
    # phi: The vector of buffered input data at step i
    phi = np.zeros(n)
    # filter weight
    w = np.zeros(n)
    # output data
    s = np.zeros(len(x))
    
    # フィルタリング
    for i, x_i in enumerate(x):
        # buffering
        phi[1:] = phi[0:-1]
        phi[0] = x_i

        # e: error
        e = d[i] - np.dot(w, phi)
        
        # filter update
        w = w + mu * e / (0.01 + np.dot(phi, phi)) * phi

        # output
        s[i] = e

    return s

# **********************************************
# 適応フィルタによるノイズ除去のサンプル
# **********************************************
def denoising_sample():
    from scipy import signal

    # -------------------------------------
    # パラメータの設定
    # -------------------------------------
    # 時間の最大値
    max_time = 1
    # 時間幅(サンプリングレート)
    dt = 0.001

    # -------------------------------------
    # 信号の生成
    # -------------------------------------
    # 時間のリスト
    time_array = np.arange(0, max_time, dt)

    # ノイズ
    noise = 0.3 * np.random.randn(len(time_array))

    # ノイズにLPFを適用
    fn = 1 / (dt * 2)
    n, wn = signal.buttord(300 / fn, 500 / fn, 1, 40)
    b, a = signal.butter(n, wn, 'low')
    noise_filtered = signal.filtfilt(b, a, noise)

    # サンプル信号の生成
    data = np.sin(2 * np.pi * 5 * time_array) + 1.0 * noise_filtered

    # -------------------------------------
    # 適応フィルタによるノイズ除去
    # -------------------------------------
    signal = nlms(data, noise, 64, 0.1)

    # -------------------------------------
    # 描画
    # -------------------------------------
    plt.plot(data)
    plt.plot(signal)
    plt.show()

def main():
    # サンプルコードの実行
    denoising_sample()

if __name__ == '__main__':
    main()
