from typing import Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from hmmlearn import hmm
from hmmquant import __version__, utils
from pandas.core.frame import DataFrame
from scipy.stats import kstest
from scipy.stats.stats import mode


def test_version():
    assert __version__ == "0.1.0"


def test_calc_backtest_params():

    # 举例说明，输出
    # (0, 6, 6) (0, 12, 7) (0, 18, 13) ...
    # 三元组分别代表：切片头、切片尾、expanding 参数
    # 回测函数在第一个进程输入是 0 1 2 3 4 5，expanding 会取前 6 个数，只用前 5 个数训练
    # 并把预测结果记录在 5，由于数据长度为 6，工作结束
    # 结果输出 [r_5]，代表 5 号的回测收益率

    # 回测函数在第二个进程输入是 0 1 ... 5 6 ... 11，expanding 会取前 7 个数，只用前 6 个数序训练
    # 并把预测结果记录在 6，而后 expanding 会取前 8 个数，只用前 7 个数训练 ...
    # ... 直到 expanding 取前 12 个数，只用前 11 个数序训练，并把预测结果记录在 11，工作结束
    # 结果输出 [r_6, r_7, ..., r_11]

    # 最后一个工作 (0, 84, 79)，输出 [r_78, r_79, r_80]
    # 然后拼接所有进程的结果得到回测收益率序列 [r_5, r_6, r_7, ..., r_80]
    all_data = list(range(81))
    backtest_params = utils.calc_backtest_params(len(all_data), train_min_len=5)
    assert backtest_params == [
        (0, 6, 6),
        (0, 12, 7),
        (0, 18, 13),
        (0, 24, 19),
        (0, 30, 25),
        (0, 36, 31),
        (0, 42, 37),
        (0, 48, 43),
        (0, 54, 49),
        (0, 60, 55),
        (0, 66, 61),
        (0, 72, 67),
        (0, 78, 73),
        (0, 84, 79),
    ]


def test_calc_backtest_params2():
    all_data = list(range(81))
    backtest_params = utils.calc_backtest_params2(len(all_data), train_min_len=5)
    assert backtest_params == [
        (0, 7, 6),
        (0, 9, 8),
        (0, 11, 10),
        (0, 13, 12),
        (0, 15, 14),
        (0, 17, 16),
        (0, 19, 18),
        (0, 21, 20),
        (0, 23, 22),
        (0, 25, 24),
        (0, 27, 26),
        (0, 29, 28),
        (0, 31, 30),
        (0, 33, 32),
        (0, 35, 34),
        (0, 37, 36),
        (0, 39, 38),
        (0, 41, 40),
        (0, 43, 42),
        (0, 45, 44),
        (0, 47, 46),
        (0, 49, 48),
        (0, 51, 50),
        (0, 53, 52),
        (0, 55, 54),
        (0, 57, 56),
        (0, 59, 58),
        (0, 61, 60),
        (0, 63, 62),
        (0, 65, 64),
        (0, 67, 66),
        (0, 69, 68),
        (0, 71, 70),
        (0, 73, 72),
        (0, 75, 74),
        (0, 77, 76),
        (0, 79, 78),
        (0, 81, 80),
    ]
