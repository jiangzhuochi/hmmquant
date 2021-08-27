import collections
from functools import partial
from multiprocessing import Pool
from typing import Literal, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm

from hmmquant import model, utils
from hmmquant.data_proc import INDICATOR
from hmmquant.utils import CPUs

LOGRR = INDICATOR["LOGRR"]
close_se = INDICATOR["close_se"]


def peek(all_data, state_num, **_):
    """看一下分层情况"""

    train_np = utils.normalization(all_data, plus=2).values
    # print("+++++++++++++++++")
    # print(train_np)

    m = model.run_model(train_np, state_num)
    print("训练后")
    print(m.means_)
    print(m.covars_)
    logprob, state = m.decode(train_np, algorithm="viterbi")
    start, *_, _, end = all_data.index

    r = utils.get_all_state_rr(LOGRR[start:end], state)
    print(r)
    state_group = model.distinguish_state(r, 1, 1)

    utils.draw_layered(r, name="temp")
    plt.close()
    print(collections.Counter(state))
    print(state_group)
    utils.draw_scatter(state, state_group, index_date=all_data.index, close=close_se)
    plt.close()
    raise


def calc_next_rr(
    _d: Union[pd.DataFrame, pd.Series], *, state_num: int, m: Optional[hmm.GaussianHMM]
):
    """计算模型并回测的 dirty work
    如果调用方提供了模型 m，则本函数内不估计

    注意用到了变量 LOGRR 用来取收益率"""
    "一维的话 _d 是一个 Series"

    start, *_, second_end, end = _d.index
    print(end)
    # 切片，做训练集
    train = _d[:-1]
    train_np = utils.normalization(train, plus=2).values  # type:ignore

    if m is None:
        m = model.run_model(train_np, state_num)
    # 这里的m是之前用最初的训练集估计的模型，用来解码序列
    _, state = m.decode(train_np, algorithm="viterbi")
    # 只关心当前最后一个隐藏状态，我们用它来预测下一个状态是属于涨组还是跌组
    last_state = state[-1]

    # 注意，用标签切片是前闭后闭的，使用second_end
    r = utils.get_all_state_rr(LOGRR[start:second_end], state)
    state_group = model.distinguish_state(r, 1, 1)

    if last_state in state_group.rise_state:
        return LOGRR[end]
    elif last_state in state_group.fall_state:
        return -LOGRR[end]
    else:
        return 0


def _backtest_inner(
    start,
    end,
    num,
    *,
    data: pd.DataFrame,
    state_num: int,
):
    """根据 is_estimate_once 的真假做出不同的行为
    True -> 在本函数估计一次
    False -> 在 calc_next_rr 里每次都估计"""

    # 注意 num 的数量是 训练集个数 + 1 下面要减去
    train = data[start : start + num - 1]
    train_np = utils.normalization(train, plus=2).values  # type:ignore
    m = model.run_model(train_np, state_num)

    # local_data 为该组全部数据
    local_data = data[start:end]
    # num 是训练集长度 + 1
    # len(local_data) 是整个组的长度
    ret = []
    for current in range(num, len(local_data) + 1):
        # 例如 num=161 则送进去 0 (含) 至 160 (含)
        # 用 0 (含) 至 159 (含) 解码 160 是预测
        r = calc_next_rr(local_data[:current], state_num=state_num, m=m)  # type: ignore
        ret.append(r)

    # 例如 num=161 则从 160 位置开始是回测的结果
    return pd.Series(ret, index=local_data[num - 1 :].index)  # type: ignore


def backtest(
    *,
    all_data: pd.DataFrame,
    state_num: int,
    train_min_len: int,
    every_group_len: int,
    return_indicator: Literal["yearr", "sharpe", "maxdd"] = "yearr",
):
    """every_group_len 表明间隔 every_group_len 估计一次模型"""

    strategy_name = (
        f"{'-'.join(all_data.columns)}"
        if isinstance(all_data, pd.DataFrame)
        else f"{all_data.name}",  # type:ignore
        f"{state_num},{train_min_len},{every_group_len}",
    )

    with Pool(processes=CPUs) as p:
        rr = pd.concat(
            p.starmap(
                partial(
                    _backtest_inner,
                    data=all_data,
                    state_num=state_num,
                ),
                utils.calc_backtest_params2(
                    data_len=len(all_data),
                    train_min_len=train_min_len,
                    every_group_len=every_group_len,
                ),
            )
        )
    contrast_rr = pd.DataFrame(
        {
            "strategy": rr,
            "+bench": LOGRR[rr.index[0] : rr.index[-1]],
            "-bench": -LOGRR[rr.index[0] : rr.index[-1]],
        }
    )
    utils.draw_layered(contrast_rr, name=strategy_name)
    evaluation = utils.get_evaluation(contrast_rr, 0.00, name=strategy_name)
    return evaluation.loc["strategy", return_indicator]
