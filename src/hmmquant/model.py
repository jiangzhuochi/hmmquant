from collections import namedtuple

import numpy as np
import pandas as pd
from hmmlearn import hmm

StateGroup = namedtuple("StateGroup", ["rise_state", "fall_state", "shock_state"])


def run_model(training_set: np.ndarray, state_num: int = 3) -> hmm.GaussianHMM:

    model = hmm.GaussianHMM(
        n_components=state_num,
        covariance_type="full",
        init_params="t",
        params="tmc",
        tol=1.0e-15,  # safe?
    )

    # 固定选择第一个状态以开始
    # 由于估计模型之前所有状态都是等价的，因此选择哪个状态开始均可
    # 给出初始值，锁
    model.startprob_ = np.append([1], np.zeros(state_num - 1))

    # every_group_num
    eg_num = len(training_set) // state_num
    _ts = sorted(training_set.flatten())
    all_group = []
    for g in range(state_num):
        if g == state_num - 1:
            all_group.append(_ts[eg_num * g :])
        else:
            all_group.append(_ts[eg_num * g : eg_num * (g + 1)])

    means_ = []
    for g in all_group:
        means_.append(np.mean(g))
    means_ = np.array(means_).reshape(-1, 1)

    covars_ = []
    for g in all_group:
        covars_.append(np.var(g))
    covars_ = np.array(covars_).reshape(-1, 1, 1)

    # print("训练前")
    # print(f"{means_=}")
    # print(f"{covars_=}")

    model.means_ = means_
    model.covars_ = covars_

    model.fit(training_set)

    return model


def distinguish_state(r: pd.DataFrame, rise_num: int, fall_num: int):
    flag = True
    while rise_num + fall_num > len(r.columns):
        if flag:
            fall_num -= 1
            flag = False
        else:
            rise_num -= 1
            flag = True
    rise_state = set(r.sum().nlargest(rise_num, keep="all").index)
    fall_state = set(r.sum().nsmallest(fall_num, keep="all").index)
    shock_state = set(r.columns) - rise_state - fall_state
    state_group = StateGroup(list(rise_state), list(fall_state), list(shock_state))
    return state_group


def distinguish_state2(r: pd.DataFrame, *_):
    rise_state = set(r.sum()[r.sum() >= 0].index)  # type: ignore
    fall_state = set(r.columns) - rise_state
    state_group = StateGroup(list(rise_state), list(fall_state), list())
    return state_group
