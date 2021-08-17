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
        all_data=MACD[-2000:],
        # 训练集序列输入方法, rolling | expanding
        method="rolling",
        # 隐含状态数
        state_num=4,
        # 对于 rolling, 则是训练集个
        # 对于 expanding, 则是最小的训练集个数
        train_min_len=1600,
    )

    # peek2(rr=LOGRR, close_param=close_se, **config)
    backtest(**config)
