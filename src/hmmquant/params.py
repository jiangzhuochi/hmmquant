from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from hmmquant.data_proc import INDICATOR

# ================ 数据全局变量
LOGRR = INDICATOR["LOGRR"]
MACD = INDICATOR["MACD"]
MA5 = INDICATOR["MA5"]
MA200 = INDICATOR["MA200"]
CCI = INDICATOR["CCI"]
RSI = INDICATOR["RSI"]
close_se = INDICATOR["close_se"]


# ================ 参数默认值
config = dict(
    # all_data=INDICATOR[["RSI", "MACD"]].iloc[-1000:, :],
    all_data=INDICATOR[["RSI"]].iloc[:, :],
    # all_data=RSI,
    # 隐含状态数
    state_num=4,
    # 训练集个数
    train_min_len=240,
    # 间隔 every_group_len 估计一次模型
    every_group_len=320,
    return_indicator="yearr",
    # output="sig"
)

# ================ 超参数设定
# 最小训练集 train_min_len 下限
tml_d = 17 * 10
# 最小训练集上限
tml_u = 17 * 35
# 最小训练集步长
tml_s = 17 * 5
# 间隔天数 every_group_len 下限
egl_d = 17 * 20
# 间隔天数上限
egl_u = 17 * 35
# 间隔天数步长
egl_s = 17 * 5
train_min_len_range = range(tml_d, tml_u, tml_s)
every_group_len_range = range(egl_d, egl_u, egl_s)

# # ================ 进度条相关
# # 回测组数
# num_of_backtest_group = len(train_min_len_range) * len(every_group_len_range)
# # 每组回测天数(不精确)
# num_of_backtest_days = len(close_se)
# # # 回测天数总计(不精确)
# # total = num_of_backtest_group * num_of_backtest_days
# overall_progress = Progress()
# one_group_progress = Progress()
# overall_task = overall_progress.add_task(
#     "All Backtest", total=int(num_of_backtest_group)
# )
# one_group_task_ls = [
#     one_group_progress.add_task(
#         f"tml={train_min_len} egl={every_group_len}",
#         total=int(num_of_backtest_days),
#     )
#     for train_min_len in train_min_len_range
#     for every_group_len in every_group_len_range
# ]
# # 构造回测组和任务id之间的映射
# desc_taskid_map = {task.description: task.id for task in one_group_progress.tasks}
# progress_table = Table.grid()
# progress_table.add_row(
#     Panel.fit(
#         overall_progress,
#         title="Overall Progress",
#         border_style="green",
#         padding=(2, 2),
#     ),
#     Panel.fit(
#         one_group_progress, title="[b]Backtest", border_style="red", padding=(2, 2)
#     ),
# )