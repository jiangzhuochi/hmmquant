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

StateGroup = namedtuple("StateGroup", ["rise_state", "fall_state", "shock_state"])

# data_df = futures_main_sina("T0")
# data_df.to_pickle("data/t0.pkl")
data = pd.read_csv("./data/data.csv", index_col=0, parse_dates=True)
c_se = data["close_30min"]
logrr = utils.get_logrr(c_se)

macd = data["MACD_12_26_30min"][25:]
logrr = logrr[macd.index]

all_data = macd

pivot = 0.93
train_rr, test_rr = utils.train_test_split(logrr, pivot)

train, test = utils.train_test_split(all_data, pivot)
# train = logrr
# print(train)
# 训练
train_np = utils.normalization(train, plus=2).values.reshape(-1, 1)  # type:ignore
state_num = 5  #  <= 隐含状态数
m = model.run_model(train_np, state_num)
logprob, state = m.decode(train_np, algorithm="viterbi")
r = utils.get_all_state_rr(train_rr, state)


# def distinguish_state(r: pd.DataFrame, rise_num: int, fall_num: int):
#     assert rise_num + fall_num <= len(r.columns)
#     rise_state = set(r.sum().nlargest(rise_num, keep="all").index)
#     fall_state = set(r.sum().nsmallest(fall_num, keep="all").index)
#     shock_state = set(r.columns) - rise_state - fall_state
#     state_group = StateGroup(list(rise_state), list(fall_state), list(shock_state))
#     return state_group


def distinguish_state(r: pd.DataFrame):
    rise_state = set(r.sum()[r.sum() >= 0].index)  # type: ignore
    fall_state = set(r.columns) - rise_state
    state_group = StateGroup(list(rise_state), list(fall_state), list())
    return state_group


# # utils.draw_img(r)
# plt.show()
# plt.close()
print(collections.Counter(state))
state_group = distinguish_state(r)
print(state_group)

# # 散点图

# color_list = []
# for s in state:
#     if s in state_group.rise_state:
#         color_list.append("red")
#     elif s in state_group.fall_state:
#         color_list.append("green")
#     else:
#         color_list.append("gray")
# print(color_list)
# ax = plt.subplot()
# ax.scatter(train.index, c_se[1:len(train)+1].values, color=color_list,s=10)
# plt.show()


# # test
test = test[:100]
# 用现在的时间对应下一个时间的收益率，留作后用
# logrrnext = logrr.shift(-1)
# 存回测结果的字典
ret = {}
for d, v in test.iteritems():
    len_ = 1
    # print(pd.Series({d: v}))
    # 将测试集的元素依次加到训练集
    # 在这里，标准化观测序列
    train_np = utils.normalization(train, plus=2).values.reshape(-1, 1)  # type:ignore
    # 这里的m是之前用最初的训练集估计的模型，用来解码序列
    logprob, state = m.decode(train_np, algorithm="viterbi")
    # 只关心当前最后一个隐藏状态，我们用它来预测下一个状态是属于涨组还是跌组
    last_state = state[-1]

    print(f"{last_state=}")
    # 将m的某个状态转移到涨组和跌组状态的概率算出
    transmat_ = pd.DataFrame(m.transmat_)
    # 方向映射表
    direction_map = pd.DataFrame(
        {
            "rise_p": transmat_[state_group.rise_state].sum(axis=1),
            "fall_p": transmat_[state_group.fall_state].sum(axis=1),
        }
    )
    # print(direction_map)
    # rise_p 涨， fall_p 跌
    direction = direction_map.loc[last_state].idxmax()  # type: ignore
    # print(direction)
    if direction == "rise_p":
        ret[d] = logrr[d]  # type: ignore
    elif direction == "fall_p":
        ret[d] = -logrr[d]  # type: ignore
    # print(ret)
    train = train.append(pd.Series({d: v}))

pd.Series(ret).cumsum().plot(label='strategy')
logrr[test.index[0] : test.index[-1]].cumsum().plot(label='bench')
plt.legend()
plt.show()
# r = utils.get_all_state_rr(test_rr[-len_ - 1 :], state[-len_ - 1 :])

# print(r)
# # m = model.run_model(train_np, state_num)


# utils.draw_img(r)
# plt.show()
