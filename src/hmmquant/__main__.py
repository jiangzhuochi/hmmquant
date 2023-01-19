import pandas as pd
from rich.live import Live

from hmmquant import utils
from hmmquant.backtest import backtest, peek
from hmmquant.data_proc import INDICATOR
from hmmquant.params import *

# ================ 数据全局变量
LOGRR = INDICATOR["LOGRR"]
MACD = INDICATOR["MACD"]
MA5 = INDICATOR["MA5"]
MA200 = INDICATOR["MA200"]
CCI = INDICATOR["CCI"]
RSI = INDICATOR["RSI"]
close_se = INDICATOR["close_se"]


def main():
    # ================ 回测循环体
    # with Live(progress_table, refresh_per_second=10):
    tml_dict = {}
    for train_min_len in train_min_len_range:
        egl_dict = {}
        for every_group_len in every_group_len_range:
            # 回测组名称
            print(f"start: {train_min_len=} {every_group_len=} ")
            config.update(
                dict(
                    train_min_len=train_min_len,
                    every_group_len=every_group_len,
                )
            )
            # # 看一下隐状态散点图, 会终止回测
            # peek(**config)
            ret = backtest(**config)  # type:ignore
            egl_dict[every_group_len] = ret

            # # 更新进度条
            # overall_progress.advance(overall_task)

        tml_dict[train_min_len] = egl_dict

    # ================ 绘制回测表现热力图
    backtest_df = pd.DataFrame(tml_dict)
    backtest_df.index.name = "every_group_len"
    backtest_df.columns.name = "train_min_len"
    utils.draw_heatmap(
        backtest_df,
        name=(
            f"{'-'.join(config['all_data'].columns)}"  # type:ignore
            if isinstance(config["all_data"], pd.DataFrame)
            else f"{config['all_data'].name}",  # type:ignore
            f"{config['state_num']}{train_min_len_range}{every_group_len_range}",  # type:ignore
        ),
    )


if __name__ == "__main__":
    main()
