import collections
from collections import namedtuple
from typing import Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn import hmm
from scipy import stats
from scipy.stats import kstest

from hmmquant import model, utils

# data_df = futures_main_sina("T0")
# data_df.to_pickle("data/t0.pkl")
data = pd.read_csv("./data/data.csv", index_col=0, parse_dates=True)
c_se = data["close_30min"]
logrr = utils.get_logrr(c_se)

macd = data["MACD_12_26_30min"][25:]
logrr = logrr[macd.index]

all_data = macd
pivot = 0.458


############## 绘图 #############

train_rr, test_rr = utils.train_test_split(logrr, pivot)
train, test = utils.train_test_split(all_data, pivot)
# train = logrr
# print(train)
# 训练

train_np = utils.normalization(train, plus=2).values.reshape(-1, 1)  # type:ignore
state_num = 4 #  <= 隐含状态数
# 在该条件下，状态 6 个以上效果不好
# 状态越多，对小变动越敏感，经常出现多空切换
# 状态越少，对大趋势把握更准确，但是对市场更迟钝
m = model.run_model(train_np, state_num)
print(m.means_)
logprob, state = m.decode(train_np, algorithm="viterbi")
r = utils.get_all_state_rr(train_rr, state)
print(r)
state_group = model.distinguish_state(r, 1, 1)
# state_group2 = model.distinguish_state2(r)

utils.draw_layered(r)
plt.close()
print(collections.Counter(state))
print(state_group)
utils.draw_scatter(state, state_group, index_date=train.index, close=c_se)
plt.close()
##################################


# # test
# test = test[:100]
# 用现在的时间对应下一个时间的收益率，留作后用
# logrrnext = logrr.shift(-1)
# 存回测结果的字典
ret = {}


def calc_next_rr(_d, *, state_num=4):
    "一维的话 _d 是一个 Series"

    start, *_, second_end, end = _d.index
    print(end)
    # 切片，做训练集
    train = _d[:-1]
    train_np = utils.normalization(train, plus=2).values.reshape(  # type:ignore
        -1, 1
    )
    m = model.run_model(train_np, state_num)
    # 这里的m是之前用最初的训练集估计的模型，用来解码序列
    _, state = m.decode(train_np, algorithm="viterbi")
    # 只关心当前最后一个隐藏状态，我们用它来预测下一个状态是属于涨组还是跌组
    last_state = state[-1]

    # print(f"{last_state=}")
    # 将m的某个状态转移到涨组和跌组状态的概率算出
    transmat_ = pd.DataFrame(m.transmat_)
    # 方向映射表

    # 注意，用标签切片是前闭后闭的，使用second_end
    r = utils.get_all_state_rr(logrr[start:second_end], state)
    state_group = model.distinguish_state(r, 1, 1)
    # direction_map = pd.DataFrame(
    #     {
    #         "rise_p": transmat_[state_group.rise_state].sum(axis=1),
    #         "fall_p": transmat_[state_group.fall_state].sum(axis=1),
    #     }
    # )
    # # print(direction_map)
    # # rise_p 涨， fall_p 跌
    # direction = direction_map.loc[last_state].idxmax()  # type: ignore
    # # print(direction)
    # if direction == "rise_p":
    #     return logrr[end]
    # elif direction == "fall_p":
    #     return -logrr[end]
    if last_state in state_group.rise_state:
        return logrr[end]
    elif last_state in state_group.fall_state:
        return -logrr[end]
    else:
        return 0


rr = all_data[:].expanding(3001).apply(calc_next_rr).dropna()
# print(rr)

contrast_rr = pd.DataFrame(
    {
        "strategy": rr,
        "+bench": logrr[rr.index[0] : rr.index[-1]],
        "-bench": -logrr[rr.index[0] : rr.index[-1]],
    }
)
utils.draw_layered(contrast_rr)
print(contrast_rr)
plt.show()


##############

# run_model_every = 100
# flag = run_model_every
# for d, v in test.iteritems():
#     len_ = 1
#     # print(pd.Series({d: v}))
#     # 在这里，标准化观测序列
#     train_np = utils.normalization(train, plus=2).values.reshape(-1, 1)  # type:ignore
#     # 这里的m是之前用最初的训练集估计的模型，用来解码序列
#     logprob, state = m.decode(train_np, algorithm="viterbi")
#     # 只关心当前最后一个隐藏状态，我们用它来预测下一个状态是属于涨组还是跌组
#     last_state = state[-1]

#     # print(f"{last_state=}")
#     # 将m的某个状态转移到涨组和跌组状态的概率算出
#     transmat_ = pd.DataFrame(m.transmat_)
#     # 方向映射表
#     direction_map = pd.DataFrame(
#         {
#             "rise_p": transmat_[state_group.rise_state].sum(axis=1),
#             "fall_p": transmat_[state_group.fall_state].sum(axis=1),
#         }
#     )
#     # print(direction_map)
#     # rise_p 涨， fall_p 跌
#     direction = direction_map.loc[last_state].idxmax()  # type: ignore
#     # print(direction)
#     if direction == "rise_p":
#         ret[d] = logrr[d]  # type: ignore
#     elif direction == "fall_p":
#         ret[d] = -logrr[d]  # type: ignore
#     # print(ret)
#     # print(pd.Series({d: v}))
#     train = train.append(pd.Series({d: v}))
#     print(f"{len(train)=}")

#     # 每隔run_model_every重新训练
#     flag -= 1
#     if flag == 0:
#         train_np = utils.normalization(train, plus=2).values.reshape(  # type:ignore
#             -1, 1
#         )
#         m = model.run_model(train_np, state_num)
#         flag = run_model_every


# ret_rr = pd.DataFrame(
#     {
#         "strategy": pd.Series(ret),
#         "+bench": logrr[test.index[0] : test.index[-1]],
#         "-bench": -logrr[test.index[0] : test.index[-1]],
#     }
# )
# utils.draw_layered(ret_rr)
# # pd.Series(ret).cumsum().plot(label="strategy")
# # logrr[test.index[0] : test.index[-1]].cumsum().plot(label="bench")
# # plt.legend()
# plt.show()
# # r = utils.get_all_state_rr(test_rr[-len_ - 1 :], state[-len_ - 1 :])

# # print(r)
# # # m = model.run_model(train_np, state_num)


# # utils.draw_img(r)
# # plt.show()
