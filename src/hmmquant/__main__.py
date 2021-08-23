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
        # all_data=INDICATOR[["RSI", "MACD"]].iloc[-1000:, :],
        all_data=INDICATOR[["RSI",]].iloc[:, :],
        # all_data=RSI,
        # 训练集序列输入方法 只允许 None 后面删掉
        method=None,
        # 隐含状态数
        state_num=4,
        # 训练集个数
        train_min_len=240,
        # 间隔 every_group_len 估计一次模型
        every_group_len=320,
        return_indicator="yearr",
    )

    train_min_len_range = range(16 * 20, 16 * 40, 16 * 500)
    every_group_len_range = range(16 * 20, 16 * 40, 16 * 500)
    grid_search_name = (
        f"{'-'.join(config['all_data'].columns)}"
        if isinstance(config["all_data"], pd.DataFrame)
        else f"{config['all_data'].name}",  # type:ignore
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
