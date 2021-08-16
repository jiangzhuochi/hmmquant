import collections
import math
import os
from collections import namedtuple
from typing import Literal, Tuple, Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
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


def evaluation(
    rr: Union[pd.Series, pd.DataFrame], risk_free_rr: float, bench=False
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
    if bench:
        local_rr = rr[["Strategy", "Benchmark"]]
        cum_rr = local_rr.cumsum()
    else:
        local_rr = rr
        cum_rr = local_rr.cumsum()
    # 累计收益
    cr = cum_rr.iloc[-1]
    # 年化收益
    yearr = (1 + cr) ** (250 / len(rr)) - 1
    # 年化波动率
    yearstd = local_rr.std() * 250 ** 0.5
    # 夏普比率
    sharpe = (yearr - risk_free_rr) / yearstd  # type: ignore
    # 最大回撤
    if bench:
        max_drawdown = (cum_rr.expanding().max() - cum_rr).apply(max)  # type: ignore
    else:
        max_drawdown = max(cum_rr.expanding().max() - cum_rr)  # type: ignore
    # df = pd.concat([cr, yearr, yearstd, sharpe, max_drawdown], axis=1)  # type: ignore
    # df.columns = ["累计收益", "年化收益", "年化波动率", "夏普比率", "最大回撤"]
    df = pd.DataFrame(
        [[cr, yearr, yearstd, sharpe, max_drawdown]],
        columns=["累计收益", "年化收益", "年化波动率", "夏普比率", "最大回撤"],
    )
    return df


def calc_backtest_params(
    data_len: int,
    train_min_len: int,
    method: Literal["expanding", "rolling"] = "expanding",
):
    # 使用 calc_backtest_params2 
    # 先只写 method = "expanding" 的情况

    assert CPUs is not None
    # 最大分组个数
    group_num = 5 * CPUs
    # 每组数据个数，向上取整
    every_group_len = math.ceil(data_len / group_num)
    # 当每组数据个数 不大于 训练集要求最小的个数时 ...
    while every_group_len <= train_min_len:
        # ... 减少分组个数
        group_num -= 1
        if group_num == 0:
            raise Exception("训练数据不足")
        every_group_len = math.ceil(data_len / group_num)

    # 最小训练长度会限制第一个组的大小，再分组上还可以改进
    # 例如在总数中单独把第一组划分出来，其余的等分就好了
    # 写好了才想起来，于是写了 calc_backtest_params2，更好

    # 我笔记本配置如下
    # MacBook Air (Retina, 13-inch, 2020)
    # 1.1 GHz 四核Intel Core i5
    # 8 GB 3733 MHz LPDDR4X
    # 一个进程吃 CPU 65% 左右
    # 两个进程吃满

    ends = [*range(every_group_len, data_len + every_group_len, every_group_len)]
    # train_min_len + 1 是为了将最后一个日期提取出来
    # 便于索引收益率，记录模型对该天的涨跌判断
    # 训练只用 train_min_len 个
    nums = [train_min_len + 1, *map(lambda i: i + 1, ends[:-1])]
    starts = [0 for _ in range(len(ends))]
    return list(zip(starts, ends, nums))


def calc_backtest_params2(
    data_len: int,
    train_min_len: int,
    method: Literal["expanding", "rolling"] = "expanding",
):
    # 先只写 method = "expanding" 的情况
    # todo: rolling
    # 显然 rolling 的回测速度会快得多

    assert CPUs is not None
    group_num = 5 * CPUs
    if data_len <= train_min_len:
        raise Exception("训练数据不足")
    every_group_len = math.ceil((data_len - train_min_len) / group_num)

    ends = [
        *range(
            train_min_len + every_group_len, data_len + every_group_len, every_group_len
        )
    ]
    nums = [train_min_len + 1, *map(lambda i: i + 1, ends[:-1])]
    starts = [0 for _ in range(len(ends))]
    return list(zip(starts, ends, nums))


# https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
def ewma_vectorized_safe(data, alpha, row_size=None, dtype=None, order="C", out=None):
    """
    Reshapes data before calculating EWMA, then iterates once over the rows
    to calculate the offset without precision issues
    :param data: Input data, will be flattened.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param row_size: int, optional
        The row size to use in the computation. High row sizes need higher precision,
        low values will impact performance. The optimal value depends on the
        platform and the alpha being used. Higher alpha values require lower
        row size. Default depends on dtype.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    :return: The flattened result.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = float
    else:
        dtype = np.dtype(dtype)

    if row_size is not None:
        row_size = int(row_size)
    else:
        row_size = get_max_row_size(alpha, dtype)

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    ewma_vectorized_2d(
        data_main_view,
        alpha,
        axis=1,
        offset=0,
        dtype=dtype,
        order="C",
        out=out_main_view,
    )

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        ewma_vectorized(
            data[-trailing_n:],
            alpha,
            offset=out_main_view[-1, -1],
            dtype=dtype,
            order="C",
            out=out[-trailing_n:],
        )
    return out


def get_max_row_size(alpha, dtype=float):
    assert 0.0 <= alpha < 1.0
    # This will return the maximum row size possible on
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use np.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = np.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(np.log(epsilon) / np.log(1 - alpha)) + 1


def ewma_vectorized(data, alpha, offset=None, dtype=None, order="C", out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    scaling_factors = np.power(
        1.0 - alpha, np.arange(data.size + 1, dtype=dtype), dtype=dtype
    )
    # create cumulative sum array
    np.multiply(
        data, (alpha * scaling_factors[-2]) / scaling_factors[:-1], dtype=dtype, out=out
    )
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out


def ewma_vectorized_2d(
    data, alpha, axis=None, offset=None, dtype=None, order="C", out=None
):
    """
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order, out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(
        1.0 - alpha, np.arange(row_size + 1, dtype=dtype), dtype=dtype
    )
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(
            alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype), dtype=dtype
        )
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype,
        out=out_view,
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out
