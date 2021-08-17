from functools import reduce
from pathlib import Path
from typing import Iterable, Union

import matplotlib.pyplot as plt
import pandas as pd

IMG_DIR = Path(".") / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)


def draw_layered(rr_df: pd.DataFrame, name: Union[str, Iterable[str]]):
    """画分层图，用于展示不同状态的走势
    rr_df 列是不同状态的收益率
    name 是保存的文件名
    """
    index_date = rr_df.index
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(16, 9))
    index_x = list(range(len(index_date)))
    pretty_x = index_x[:: len(index_x) // 20]
    pretty_date = index_date[:: len(index_date) // 20]
    ax.set_xticks(pretty_x)
    ax.set_xticklabels(pretty_date, rotation=30)
    ax.plot(index_x, rr_df.cumsum().values, label=rr_df.columns)
    ax.legend()
    if isinstance(name, str):
        fig.savefig(IMG_DIR / f"{name}.png", dpi=300)
    else:
        # 可迭代的
        pathname = reduce(lambda x, y: x / y, name, IMG_DIR)
        pathname.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pathname, dpi=300)


def draw_scatter(state, state_group, index_date, close):
    """画散点图，用于在价格走势上展示隐藏状态

    state 状态序列
    state_group 状态所在的组，涨还是跌
    index_date 要画图的时间序列索引
    close 价格走势，时间跨度要包含index_date的跨度
    """
    color_list = []
    for s in state:
        if s in state_group.rise_state:
            color_list.append("red")
        elif s in state_group.fall_state:
            color_list.append("green")
        else:
            color_list.append("gray")

    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(16, 9))
    index_x = list(range(len(index_date)))
    pretty_x = index_x[:: len(index_x) // 20]
    pretty_date = index_date[:: len(index_date) // 20]
    ax.set_xticks(pretty_x)
    ax.set_xticklabels(pretty_date, rotation=30)
    ax.scatter(
        index_x, close[1 : len(index_date) + 1].values, color=color_list, s=7  # type: ignore
    )
    plt.show()
    fig.savefig(IMG_DIR / "scatter.png", dpi=300)
