import collections
from collections import namedtuple
from typing import Union, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from hmmlearn import hmm
from pandas.core.frame import DataFrame
from scipy import stats
from scipy.stats import kstest

from hmmquant import model, utils

StateGroup = namedtuple("StateGroup", ["rise_state", "fall_state", "shock_state"])

# data_df = futures_main_sina("T0")
# data_df.to_pickle("data/t0.pkl")
data = pd.read_csv("./data/data.csv", index_col=0, parse_dates=True)
c_se = data["close_30min"]
logrr = utils.get_logrr(c_se)

train, verify, test = utils.train_verify_test_split(logrr)
# train = logrr
# 训练
train_np = utils.normalization(train, plus=2).values.reshape(-1, 1)  # type:ignore
state_num = 4  #  <= 隐含状态数
m = model.run_model(train_np, state_num)
logprob, state = m.decode(train_np, algorithm="viterbi")
r = utils.get_all_state_rr(train, state)

utils.draw_img(r)
print(collections.Counter(state))


def distinguish_state(r: pd.DataFrame, rise_num: int, fall_num: int):
    assert rise_num + fall_num <= len(r.columns)
    rise_state = set(r.sum().nlargest(rise_num, keep="all").index)
    fall_state = set(r.sum().nsmallest(fall_num, keep="all").index)
    shock_state = set(r.columns) - rise_state - fall_state
    state_group = StateGroup(list(rise_state), list(fall_state), list(shock_state))
    return state_group


state_group = distinguish_state(r, 2, 2)
transmat_ = pd.DataFrame(m.transmat_)
a = pd.DataFrame(
    {
        "rise_p": transmat_[state_group.rise_state].sum(axis=1),
        "fall_p": transmat_[state_group.fall_state].sum(axis=1),
        "shock_p": transmat_[state_group.shock_state].sum(axis=1),
    }
)
print(a)
plt.show()

# # 验证
# verify = train.append(verify[:1])
# print(verify)
# verify_np = utils.normalization(verify, plus=2).values.reshape(-1, 1)  # type:ignore
# logprob, state = m.decode(verify_np, algorithm="viterbi")
# print(m.transmat_)
# print(state)
# r = utils.get_all_state_rr(verify, state)


# utils.draw_img(r)
# print(collections.Counter(state))

# StateGroup = namedtuple("StateGroup", ["rise_state", "fall_state", "shock_state"])

# _n = state_num // 3
# # r.sum().nlargest(_n,keep='all'), r.sum().nsmallest(_n,keep='all'),
# rise_state = set(r.sum().nlargest(_n, keep="all").index)
# fall_state = set(r.sum().nsmallest(_n, keep="all").index)
# shock_state = set(r.columns) - rise_state - fall_state
# state_group = StateGroup(rise_state, fall_state, shock_state)
# print(state_group)

# plt.show()
# # test_np = utils.normalization(test, plus=2).values.reshape(-1, 1)  # type:ignore
# # logprob, state = m.decode(test_np, algorithm="viterbi")
# # r = utils.get_all_state_rr(test, state)
