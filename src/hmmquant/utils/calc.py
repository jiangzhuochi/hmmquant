import math
import os
from collections import namedtuple
from functools import reduce, wraps
from pathlib import Path
from typing import Literal, Optional, Union, overload

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from scipy import stats

from .draw import save_with_root

CSV_DIR = Path(".") / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

CPUs = os.cpu_count()
if CPUs is None:
    CPUs = 4


def normalization(raw_data: Union[pd.Series, pd.DataFrame], plus=2) -> pd.DataFrame:
    def _norm(se: pd.Series):
        # z_score标准化
        mean, std = se.describe()[["mean", "std"]]
        z_score_scaling = (se - mean) / std

        # minmax标准化
        ma, mi = z_score_scaling.describe()[["max", "min"]]
        min_max_scaling = (z_score_scaling - mi) / (ma - mi) + plus

        # 使用boxcox
        boxcoxed_data, _ = stats.boxcox(min_max_scaling)  # type: ignore
        return pd.Series(boxcoxed_data, index=se.index)

    # print(raw_data)
    raw_data = pd.DataFrame(raw_data)
    # print(raw_data)
    if isinstance(raw_data, pd.Series):
        return pd.DataFrame(_norm(raw_data))
    else:
        return raw_data.apply(_norm)


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


@save_with_root(CSV_DIR)
def get_evaluation(
    rr: Union[pd.Series, pd.DataFrame], risk_free_rr: float, name
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

    df: DataFrame = pd.concat([cr, yearr, yearstd, sharpe, max_drawdown], axis=1)  # type: ignore
    # ["累计收益", "年化收益", "年化波动率", "收益波动比", "最大回撤"]
    df.columns = ["cr", "yearr", "yearstd", "sharpe", "maxdd"]

    df.to_csv(f"{name}.csv")
    return df


def calc_backtest_params2(
    data_len: int,
    train_min_len: int,
    every_group_len: int,
):
    """计算多进程回测需要的参数
    every_group_len 表明间隔 every_group_len 估计一次模型"""

    assert CPUs is not None
    if data_len <= train_min_len:
        raise Exception("训练数据不足")

    ends = [
        *range(
            train_min_len + every_group_len, data_len + every_group_len, every_group_len
        )
    ]
    nums = [train_min_len + 1 for _ in range(len(ends))]
    starts = [0, *map(lambda i: i - train_min_len, ends[:-1])]
    return list(zip(starts, ends, nums))
