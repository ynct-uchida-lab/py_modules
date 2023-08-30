# py_modules

## Overview

pythonでよく使う機能を関数やクラスにまとめたもの

## Requirement

- python3

- 必要に応じて各種モジュールが必要になる
- 詳しくは[requirments.txt](https://github.com/ynct-uchida-lab/py_modules/blob/master/requirements.txt)を参照
    - numpy
    - matplotlib
    - nptdms
    - scipy
    - etc.

## Install

- git cloneで任意のディレクトリへクローンしてくる
```sh
$ git clone https://github.com/ynct-uchida-lab/py_modules.git
```

## Usage

#### 1. パスを追加

- 例: プロジェクトのディレクトリに格納した場合
    - 以下のディレクトリ構造かつ対象のスクリプトがmain.pyの場合
    - 相対パスでパスを追加

```
.
├py_moddules
│  ├...    
│
└scripts
    └main.py
```

```python
import os
sys.path.append('../py_modules')
```

- 例: "~/workspace"にあり，プロジェクトは別ディレクトリにある場合
    - 所定のディレクトリに存在する場合は絶対パスで追加すると便利

```python
import os
from os.path import expanduser
home = expanduser("~")
workspace = os.path.join(home, 'workspace')
sys.path.append(os.path.join(workspace, 'py_modules'))
```

#### 2. インポート

- 利用したいモジュールをimport
- 例: ploter/plt_formatをインポート

```python
import ploter.plt_format as plt_format
```

## License 

Copyright (c) 2020 ynct-uchida-lab
Released under the MIT license
[LICENSE](https://github.com/ynct-uchida-lab/py_modules/blob/master/LICENSE)

## Other

内容についてのバグや要望についてはissuesへお願いします．

