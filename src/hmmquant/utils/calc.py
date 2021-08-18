import collections
import math
import os
from collections import namedtuple
from typing import Literal, Optional, Tuple, Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm
from hmmquant import model, utils
from scipy import stats
from scipy.stats import kstest

CPUs = os.cpu_count()
if CPUs is None:
    CPUs = 4


def normalization(se: pd.Series, plus=2) -> pd.Series:

    # z_score标准化
    mean, std = se.describe()[["mean", "std"]]
    z_score_scaling = (se - mean) / std

    # minmax标准化
    ma, mi = z_score_scaling.describe()[["max", "min"]]
    min_max_scaling = (z_score_scaling - mi) / (ma - mi) + plus

    # 使用boxcox
    boxcoxed_data, _ = stats.boxcox(min_max_scaling)  # type: ignore

    return pd.Series(boxcoxed_data, index=se.index)


def get_state_rr(rr_seq: pd.Series, state_seq: np.ndarray, target_s) -> pd.Series:
    """得到某个状态的收益"""
    rrlist = []
    for r, s in zip(rr_seq.values, state_seq):
        if s == target_s:
            rrlist.append(r)
        else:
            rrlist.append(0)
    return pd.Series(rrlist, index=rr_seq.index)


def get_all_state_rr(rr_seq: pd.Series, state_seq: np.ndarray) -> pd.DataFrame:
    """得到所有状态的收益"""
    _d = {}
    all_state = sorted(list(set(state_seq)))
    for s in all_state:
        _d[s] = get_state_rr(rr_seq, state_seq, s)
    return pd.DataFrame(_d)


@overload
def get_logrr(close: pd.Series) -> pd.Series:
    ...


@overload
def get_logrr(close: pd.DataFrame) -> pd.DataFrame:
    ...


def get_logrr(close: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    """计算对数收益率

    Parameters
    ----------
    close : Union[pd.Series, pd.DataFrame]
        价格

    Returns
    -------
    Union[pd.Series, pd.DataFrame]
        对数收益率
    """
    logrr = np.log(close).diff()[1:]  # type: ignore
    return logrr


def get_evaluation(
    rr: Union[pd.Series, pd.DataFrame], risk_free_rr: float = 0.0
) -> pd.DataFrame:
    """计算评价指标

    Parameters
    ----------
    rr : DataFrame
        日度收益率表，两列，分别是策略、基准
    risk_free_rr : float
        无风险收益率

    Returns
    -------
    DataFrame
        评价指标表
    """

    cum_rr = rr.cumsum()
    # 累计收益
    cr = cum_rr.iloc[-1]
    # 年化收益
    yearr = (1 + cr) ** ((250 * 16) / len(rr)) - 1
    # 年化波动率
    yearstd = rr.std() * (250 * 16) ** 0.5
    # 收益波动比
    sharpe = (yearr - risk_free_rr) / yearstd  # type: ignore
    # 最大回撤
    max_drawdown = (cum_rr.expanding().max() - cum_rr).apply(max)  # type: ignore

    df = pd.concat([cr, yearr, yearstd, sharpe, max_drawdown], axis=1)  # type: ignore
    # ["累计收益", "年化收益", "年化波动率", "收益波动比", "最大回撤"]
    df.columns = ["cr", "yearr", "yearstd", "sharpe", "maxdd"]
    return df


def calc_backtest_params2(
    data_len: int,
    train_min_len: int,
    method: Optional[Literal["expanding", "rolling"]],
    every_group_len: Optional[int],
):
    """计算多进程回测需要的参数
    如果指定了 every_group_len 则表明间隔 every_group_len 估计一次模型
    且 method 必须为 None"""

    assert CPUs is not None
    if data_len <= train_min_len:
        raise Exception("训练数据不足")

    if every_group_len is None:
        every_group_len = math.ceil((data_len - train_min_len) / (5 * CPUs))
    else:
        assert method is None

    ends = [
        *range(
            train_min_len + every_group_len, data_len + every_group_len, every_group_len
        )
    ]
    if method == "expanding":
        nums = [train_min_len + 1, *map(lambda i: i + 1, ends[:-1])]
        starts = [0 for _ in range(len(ends))]

    # method is None 时，采用 rolling 的参数
    # starts 是不断向后推进的
    elif method == "rolling" or method is None:
        nums = [train_min_len + 1 for _ in range(len(ends))]
        starts = [0, *map(lambda i: i - train_min_len, ends[:-1])]
    return list(zip(starts, ends, nums))
