import numpy as np

# データの最大値, 最小値, 平均値, 実効値を求める関数
def time_series(data):
    # 実効値を求める
    data_rms = np.sqrt(np.mean(data * data))
    # データの絶対値をとり最大値, 最小値, 平均値を求める
    data = np.abs(data)
    data_max = np.max(data)
    data_min = np.min(data)
    data_ave = np.mean(data)
    return [data_max, data_min, data_ave, data_rms]

def main():
    # サンプルデータの生成
    dt = 0.001                             # サンプリング周波数
    t = np.arange(0, 1, dt)                # 時間軸
    data = np.cos(2 * np.pi * 50 * t)
    data *= (1 + 0.5 * np.sin(2 * np.pi * 2 * t))
    data *= (1 + 0.5 * np.sin(2 * np.pi * 10 * t))

    # データの最大値, 最小値, 平均値, 実効値を表示する
    print(time_series(data))

if __name__ == "__main__":
    main()