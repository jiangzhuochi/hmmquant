from multiprocessing import Pool

from hmmquant.backtest import backtest
from hmmquant.data_proc import INDICATOR

LOGRR = INDICATOR["LOGRR"]
MACD = INDICATOR["MACD"]
close_se = INDICATOR["close_se"]


if __name__ == "__main__":

    config = dict(
        # 输入的观测序列，只支持一维
        all_data=LOGRR[-3500:],
        # 训练集序列输入方法, rolling | expanding
        method="rolling",
        # 隐含状态数
        state_num=4,
        # 对于 rolling, 则是训练集个
        # 对于 expanding, 则是最小的训练集个数
        train_min_len=3000,
    )
    backtest(**config)
