import pandas as pd

from hmmquant.backtest import backtest, peek
from hmmquant.data_proc import INDICATOR

LOGRR = INDICATOR["LOGRR"]
MACD = INDICATOR["MACD"]
MA5 = INDICATOR["MA5"]
MA200 = INDICATOR["MA200"]
CCI = INDICATOR["CCI"]
RSI = INDICATOR["RSI"]
close_se = INDICATOR["close_se"]

if __name__ == "__main__":

    config = dict(
        # 输入的观测序列，只支持一维
        # all_data=LOGRR,
        all_data=RSI,
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

    train_min_len_range = range(16 * 10, 16 * 40, 16 * 500)
    every_group_len_range = range(16 * 20, 16 * 40, 16 * 500)
    grid_search_name = (
        f"{config['all_data'].name}",  # type:ignore
        f"{config['state_num']}{train_min_len_range}{every_group_len_range}",  # type:ignore
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
            # peek(**config)
            ret = backtest(**config)

            egl_dict[every_group_len] = ret
        tml_dict[train_min_len] = egl_dict

    backtest_df = pd.DataFrame(tml_dict)
    backtest_df.index.name = "every_group_len"
    backtest_df.columns.name = "train_min_len"
    # utils.draw_heatmap(backtest_df, name=grid_search_name)
