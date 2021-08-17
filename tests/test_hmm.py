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


# def test_calc_backtest_params():

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
    # all_data = list(range(81))
    # backtest_params = utils.calc_backtest_params(len(all_data), train_min_len=5)
    # print("\n", backtest_params)
    # [
    #     (0, 6, 6),
    #     (0, 12, 7),
    #     (0, 18, 13),
    #     (0, 24, 19),
    #     (0, 30, 25),
    #     (0, 36, 31),
    #     (0, 42, 37),
    #     (0, 48, 43),
    #     (0, 54, 49),
    #     (0, 60, 55),
    #     (0, 66, 61),
    #     (0, 72, 67),
    #     (0, 78, 73),
    #     (0, 84, 79),
    # ]


def test_calc_backtest_params2():
    data_len = 200
    train_min_len = 30

    exp_params = utils.calc_backtest_params2(
        data_len, train_min_len=train_min_len, method="expanding", every_group_len=None
    )
    rol_params = utils.calc_backtest_params2(
        data_len, train_min_len=train_min_len, method="rolling", every_group_len=None
    )
    print("\n", exp_params)
    print("\n", rol_params)
