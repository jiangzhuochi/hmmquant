import collections
from functools import partial
from multiprocessing import Pool
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd

from hmmquant import model, utils
from hmmquant.data_proc import INDICATOR
from hmmquant.utils import CPUs

LOGRR = INDICATOR["LOGRR"]
close_se = INDICATOR["close_se"]


def peek(all_data, state_num):
    """看一下分层情况"""
    train_np = utils.normalization(all_data, plus=2).values.reshape(  # type:ignore
        -1, 1
    )  # type:ignore

    m = model.run_model(train_np, state_num)
    print(m.means_)
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


def calc_next_rr(_d, *, state_num):
    """计算模型并回测的 dirty work
    注意用到了变量 LOGRR 用来取收益率"""
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
    data,
    method: Literal["expanding", "rolling"] = "rolling",
    state_num: int = 4,
):
    return (
        getattr(data[start:end], method)(num)
        .apply(partial(calc_next_rr, state_num=state_num))
        .dropna()
    )


def backtest(all_data, method, state_num, train_min_len):

    with Pool(processes=CPUs) as p:
        rr = pd.concat(
            p.starmap(
                partial(
                    _backtest_inner, data=all_data, method=method, state_num=state_num
                ),
                utils.calc_backtest_params2(
                    len(all_data), train_min_len, method=method
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
    utils.draw_layered(
        contrast_rr,
        name=(str(all_data.name), str(method), f"{state_num},{train_min_len}"),
    )
