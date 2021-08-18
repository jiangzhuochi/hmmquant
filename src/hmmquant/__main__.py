from typing import Union, overload

import numpy as np
import pandas as pd
from scipy.stats import kstest

from hmmquant import model, utils
from hmmquant.backtest import backtest, peek, peek2
from hmmquant.data_proc import INDICATOR

LOGRR = INDICATOR["LOGRR"]
MACD = INDICATOR["MACD"]
MA5 = INDICATOR["MA5"]
MA200 = INDICATOR["MA200"]
CCI = INDICATOR["CCI"]
RSI = INDICATOR["RSI"]
close_se = INDICATOR["close_se"]

# data = pd.read_csv("./data/data.csv", index_col=0, parse_dates=True)
# c_se = data["close_30min"]
# logrr = utils.get_logrr(c_se)
# macd = data["MACD_12_26_30min"][25:]
# logrr = logrr[macd.index]

# all_data = macd

if __name__ == "__main__":

    config = dict(
        # 输入的观测序列，只支持一维
        all_data=CCI["2020-05-15":"2021-05-14"][-500:],
        # 训练集序列输入方法, rolling | expanding | None
        method=None,
        # 隐含状态数
        state_num=4,
        # 对于 rolling, 则是训练集个数
        # 对于 expanding, 则是最小的训练集个数
        train_min_len=240,
        # 如果指定，表示每间隔 every_group_len 估计一次模型
        every_group_len=320,
        return_indicator="yearr",
    )

    train_min_len_range = range(16 * 15, 16 * 35, 16 * 50)
    every_group_len_range = range(16 * 10, 16 * 11, 16 * 50)
    data_se: pd.Series = config["all_data"]  # type:ignore
    grid_search_name = (
        f"{data_se.name}",
        f"{train_min_len_range}{every_group_len_range}",
    )
    # train_min_len
    tml_dict = {}
    for train_min_len in train_min_len_range:
        # every_group_len
        egl_dict = {}
        for every_group_len in every_group_len_range:
            config.update(
                dict(
                    train_min_len=train_min_len,
                    every_group_len=every_group_len,
                )
            )
            # peek2(rr=LOGRR, close_param=close_se, **config)
            ret = backtest(**config)

            egl_dict[every_group_len] = ret
        tml_dict[train_min_len] = egl_dict

    backtest_df = pd.DataFrame(tml_dict)
    backtest_df.index.name = "every_group_len"
    backtest_df.columns.name = "train_min_len"
    utils.draw_heatmap(backtest_df, name=grid_search_name)
