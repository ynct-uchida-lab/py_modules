
import os
import glob
import re
import numpy as np
from nptdms import TdmsFile
from nptdms import TdmsObject
from nptdms import TdmsWriter, ChannelObject
from scipy import signal
from time import sleep

# 自然順ソートを行う
def numericalSort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# tdmsファイルのフルパスを取得し自然順ソートする
def tdms_path_get(folder):
    # フォルダ内のtdmsを検索する
    file_paths = glob.glob(folder + '/**/*.tdms', recursive=True)
    # 自然順にソート
    sorted_file_paths = sorted(file_paths, key=numericalSort)

    return sorted_file_paths

# tdmsファイルのグループ名とチャンネル名を取得
def tdms_info(tdms_file):
    # グループ名をすべて取得
    group_name = tdms_file.groups()
    # チャンネル名を取得
    channel_name = tdms_file.group_channels(group_name[0])
    # channel_name = []
    # for g in group_name:
    #     channel_name.append(tdms_file.group_channels(g))
    
    return group_name, channel_name

# tdmsを読み込み
#   numpy arrayの形式で返す
def tdms_to_np(tdms_name, time_req=False):
    # tdmsを読み込み
    tdms_file = TdmsFile(tdms_name)
    
    # チャンネルの情報を抜き出す
    group_name, channel_name = tdms_info(tdms_file)
    channel = tdms_file.object(group_name[0], channel_name[0].channel)

    # numpyに変換
    data = channel.data.astype(np.float32)

    # 時間情報が必要であれば抜き出す
    if time_req:
        time = channel.time_track()
        return data, time
    else:
        return data

# 同一フィオルダ内の複数のTDMSファイルを合わせて送り返す
def all_tdms_to_np(folder,
                   time_req=False, 
                   start_index=None, max_index=None,
                   min_size=None,
                   zero_padding=False, data_length=None):

    # フォルダ内のtdmsを検索する
    file_name = glob.glob(folder + '/*.tdms')

    # ファイル名のソート
    basename = [os.path.basename(file_name[i]) for i in range(len(file_name))]
    file_name = sorted(basename, key=numericalSort)
    # print(file_name)
    
    # データの開始インデックスの指定
    if start_index is None:
        start_index = 0

    # データの終了インデックスの指定
    if max_index is None:
        max_index = len(file_name)

    data = []
    time = []
    delete_index = []
    for i in range(start_index, max_index):
        
        tdms_name = os.path.join(folder, file_name[i])
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
        if time_req:
            d, t = tdms_to_np(tdms_name) 
            time.append(t)
        else: 
            d = tdms_to_np(tdms_name)

        # データ長を指定した長さに切る
        if data_length is not None:
            d = d[0:data_length]

        # リストにまとめる
        data.append(d)

    # データの長さを指定
    if data_length is None:
        data_length = max(map(len, data))

    # 長さがそれぞれ異なる場合は0埋めする
    if zero_padding:
        l = max(map(len, data))
        data = [np.concatenate([d, np.zeros(data_length - len(d),)]) for d in data]

    # numpyに変換
    data = np.array(data, dtype='float32')

    # 時間情報があれば時間情報も返す
    if time_req:
        # numpyに変換
        timeArr = np.array(time, dtype='float32')
        return data, time
    else:
        return data

