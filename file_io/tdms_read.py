import os
import glob
import re
import numpy as np
from nptdms import TdmsFile
# from nptdms import TdmsObject
from nptdms import TdmsWriter, ChannelObject
from scipy import signal
from time import sleep

# 自然順ソートを行う
def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# tdmsファイルのフルパスを取得し自然順ソートする
def tdms_path_get(folder):
    # フォルダ内のtdmsを検索する
    file_paths = glob.glob(folder + '/**/*.tdms', recursive=True)
    # 自然順にソート
    sorted_file_paths = sorted(file_paths, key=numerical_sort)

    return sorted_file_paths

# TDMSファイルのグループ名とチャンネル名を取得
def tdms_info(tdms_file):
    # グループ名をすべて取得
    all_groups = tdms_file.groups()
    # 先頭のグループを抜き出す
    group = all_groups[0]

    # チャンネル名を取得
    all_group_channels = group.channels()
    # 先頭のチャンネルを抜き出す
    # channel = all_group_channels[0]

    # for g in group_name:
    #     channel_name.append(tdms_file.group_channels(g))

    return all_groups, all_group_channels

# TDMSファイルを読み込みnumpy.ndarray形式で返す
def tdms_to_np(file_path, time_req=False):
    # ファイルパスからTDMSファイルを読み込み
    tdms_file = TdmsFile.read(file_path)
    
    # チャンネルの情報を抜き出す
    all_groups, all_group_channels = tdms_info(tdms_file)
    # channel = tdms_file.object(group_name[0], channel_name[0].channel)
    # channel = tdms_file[group[0]][channel[0]]
    data = all_group_channels

    # numpy.ndarrayに変換
    data = np.array(data, dtype=np.float32)
    data = data[0, :]

    # 時間情報が必要であれば抜き出す(default: None)
    #     TODO: 最新版に対応する
    time = None
    return data, time

# 同一フィオルダ内の複数のTDMSファイルを合わせて送り返す
#     time_req: TDMSに保存されている時間を取得するか(bool, default: False)
#       TODO: 要修正
#     start_index: ファイルの開始番号を指定(int, default: None)
#     max_index: ファイルの終了番号を指定(int, default: None)
#     min_size: 指定したサイズ以下のファイルを無視する(int,  default: None)
#     zero_padding: データ長がファイルごとに違う場合0埋めを行う(bool, default: False)
#     data_length: データを指定したインデックスまでにする(int, default: None)
def all_tdms_to_np(tdms_dir,
                   time_req=False, 
                   start_index=None, max_index=None,
                   min_size=None,
                   zero_padding=False, data_length=None):

    # フォルダ内のTDMSファイルのパスを検索する
    file_name = glob.glob(tdms_dir + '/*.tdms')

    # ファイル名のソート
    basename = [os.path.basename(file_name[i]) for i in range(len(file_name))]
    file_name = sorted(basename, key=numerical_sort)
    
    # ファイルの開始インデックスの指定
    if start_index is None:
        start_index = 0

    # ファイルの終了インデックスの指定
    if max_index is None:
        max_index = len(file_name)

    data = []
    time = []
    delete_index = []
    for i in range(start_index, max_index):
        
        tdms_name = os.path.join(tdms_dir, file_name[i])
        print('\rFile name: ' + tdms_name)

        # ファイルサイズのチェック
        if min_size is not None:
            # ファイルサイズの取得
            file_size = os.path.getsize(tdms_name)
            if file_size < int(min_size):
                print('file size error: ', int(i + 1))
                delete_index.append(i)
                continue

        # 時間情報の読み取り
        d, t = tdms_to_np(tdms_name, time_req=time_req) 
        time.append(t)

        # データを指定した長さに切る
        if data_length is not None:
            d = d[0: data_length]

        # リストにまとめる
        data.append(d)

    # データの長さを指定
    if data_length is None:
        data_length = max(map(len, data))

    # 長さがそれぞれ異なる場合は0埋めする
    if zero_padding:
        l = max(map(len, data))
        data = [np.concatenate([d, np.zeros(data_length - len(d),)]) for d in data]

    # numpy.ndarrayに変換
    data = np.array(data, dtype='float32')
    time = np.array(time, dtype='float32')

    return data, time

