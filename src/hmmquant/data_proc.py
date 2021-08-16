import collections
import math
import os
import time
from collections import namedtuple
from functools import partial
from multiprocessing import Pool
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
from pandas.core.series import Series
from scipy import stats

from hmmquant import model, utils
from hmmquant.utils import CPUs

data = pd.read_csv("./data/quarter.csv", index_col=0, parse_dates=True)
close_se = data["last"]
logrr = utils.get_logrr(close_se)

macd, macdsignal, macdhist = talib.MACD(  # type:ignore
    close_se, fastperiod=12, slowperiod=26, signalperiod=9
)  # type:ignore
macd: Series = macd.dropna()
################
# # 训练
# all_data = macd[-4000:]
# train_np = utils.normalization(all_data, plus=2).values.reshape(-1, 1)  # type:ignore
# state_num = 4  #  <= 隐含状态数
# # 在该条件下，状态 6 个以上效果不好
# # 状态越多，对小变动越敏感，经常出现多空切换
# # 状态越少，对大趋势把握更准确，但是对市场更迟钝
# m = model.run_model(train_np, state_num)
# print(m.means_)
# logprob, state = m.decode(train_np, algorithm="viterbi")
# start, *_, second_end, end = all_data.index

# r = utils.get_all_state_rr(logrr[start:end], state)
# print(r)
# state_group = model.distinguish_state(r, 2, 1)
# # state_group2 = model.distinguish_state2(r)

# utils.draw_layered(r)
# plt.close()
# print(collections.Counter(state))
# print(state_group)
# utils.draw_scatter(state, state_group, index_date=all_data.index, close=close_se)
# plt.close()
# ##################################


def calc_next_rr(_d, *, state_num=4):
    """模型的 dirty work
    注意用到了变量 logrr 用来取收益率"""
    "一维的话 _d 是一个 Series"

    start, *_, second_end, end = _d.index
    print(end)
    # 切片，做训练集
    train = _d[:-1]
    train_np = utils.normalization(train, plus=2).values.reshape(  # type:ignore
        -1, 1
    )
    m = model.run_model(train_np, state_num)
    # 这里的m是之前用最初的训练集估计的模型，用来解码序列
    _, state = m.decode(train_np, algorithm="viterbi")
    # 只关心当前最后一个隐藏状态，我们用它来预测下一个状态是属于涨组还是跌组
    last_state = state[-1]

    # print(f"{last_state=}")
    # 将m的某个状态转移到涨组和跌组状态的概率算出
    # 方向映射表

    # 注意，用标签切片是前闭后闭的，使用second_end
    r = utils.get_all_state_rr(logrr[start:second_end], state)
    state_group = model.distinguish_state(r, 2, 2)
    if last_state in state_group.rise_state:
        return logrr[end]

    elif last_state in state_group.fall_state:
        return -logrr[end]
    else:
        return 0


def backtest(
    start,
    end,
    num,
    *,
    data,
    method: Literal["expanding", "rolling"] = "expanding",
):
    return getattr(data[start:end], method)(num).apply(calc_next_rr).dropna()


if __name__ == "__main__":

    all_data = macd[-640:]

    params = utils.calc_backtest_params2(len(all_data), 320)

    with Pool(processes=CPUs) as p:
        rr = p.starmap(partial(backtest, data=all_data), params)

    # print(rr)
    rr = pd.concat(rr)
    contrast_rr = pd.DataFrame(
        {
            "strategy": rr,
            "+bench": logrr[rr.index[0] : rr.index[-1]],
            "-bench": -logrr[rr.index[0] : rr.index[-1]],
        }
    )
    utils.draw_layered(contrast_rr)
    print(contrast_rr)
    plt.show()
