import json
import pickle
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import talib
from hmmlearn import hmm

import hmmquant

DATA_DIR = Path(".") / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

url = (
    "https://stock2.finance.sina.com.cn/futures/api/jsonp.php"
    "?symbol=T2112&type=15=/InnerFuturesNewService.getFewMinLine"
)
res = requests.get(url)
df = (
    pd.DataFrame(json.loads(re.findall(r"\[.*?\]", res.text)[0]))
    .rename(
        columns={
            "d": "datetime",
            "o": "open",
            "h": "high",
            "l": "low",
            "c": "close",
            "v": "volume",
            "p": "position",
        }
    )
    .set_index("datetime")
    .rename(index=lambda idx: pd.to_datetime(idx))
    .astype(
        {
            **{col_name: "float64" for col_name in ["open", "high", "low", "close"]},
            **{col_name: "int64" for col_name in ["volume", "position"]},
        }
    )
)

# 如果现在的时间在 df 最后一行的时间之前
# 就抛弃最后一行
if (datetime.now() - df.index[-1]).total_seconds() < 0:
    df = df.head(-1)

"""本文件输入不断增长的 15min 数据表, 输出信号
前置条件: 15min 数据, 至少包括 OHLC"""

################################
# step 0 参数
state_num = 4
train_min_len = 170
every_group_len = 340
indicator_list = ["RSI"]

################################
# step 1 读取数据计算指标

# quarter_data = pd.read_csv(DATA_DIR / "quarter.csv", index_col=0, parse_dates=True)
quarter_data = df
close_se = quarter_data["close"]
high_se = quarter_data["high"]
low_se = quarter_data["low"]
# print(close_se)
LOGRR: pd.Series = hmmquant.utils.get_logrr(close_se)
RSI: pd.Series = talib.RSI(close_se, timeperiod=14)  # type:ignore
df = pd.DataFrame(
    {
        "close_se": close_se,
        "LOGRR": LOGRR,
        "RSI": RSI,
    }
).dropna()


################################
# step 2 模型
# 如果存在一个未过期的模型，则使用该模型
# 否则重新估计
def run_model(training_set: np.ndarray, state_num: int = 3) -> hmm.GaussianHMM:

    model = hmm.GaussianHMM(
        n_components=state_num,
        covariance_type="full",
        init_params="t",
        params="tmc",
        tol=1.0e-15,
    )

    model.startprob_ = np.append([1], np.zeros(state_num - 1))
    _ts = training_set[np.argsort(training_set[:, 0])]

    # every_group_num
    eg_num = len(training_set) // state_num
    all_group = []
    for g in range(state_num):
        if g == state_num - 1:
            all_group.append(_ts[eg_num * g :])
        else:
            all_group.append(_ts[eg_num * g : eg_num * (g + 1)])
    means_ = []
    for g in all_group:
        means_.append(np.mean(g, axis=0))
    means_ = np.array(means_)
    covars_ = []
    for g in all_group:
        covars_.append(np.cov(g, rowvar=False))
    covars_ = np.array(covars_)
    if len(covars_.shape) == 1:
        covars_ = covars_.reshape(-1, 1, 1)

    model.means_ = means_
    model.covars_ = covars_

    model.fit(training_set)
    return model


@dataclass
class ModelWithTime:
    """相当于包含模型本身和其创建时间的结构体"""

    _model: hmm.GaussianHMM
    _time: str


def create_model_with_time():

    # 取 train_min_len 个估计模型
    needed_df = df.tail(train_min_len)
    start, *_ = needed_df.index

    train = needed_df
    train_np = hmmquant.utils.normalization(train, plus=2).values  # type:ignore
    model = run_model(train_np, state_num)

    # 测试用
    # start = datetime.strptime("2021-07-20 15:15", r"%Y-%m-%d %H:%M")
    ############

    model_with_time = ModelWithTime(
        _model=model, _time=start.strftime(r"%Y-%m-%d %H:%M")
    )
    # 保存模型
    with open(DATA_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_with_time, f)
    return model_with_time


if (model_path := DATA_DIR / "model.pkl").exists():
    with open(model_path, "rb") as f:
        model_with_time = pickle.load(f)

    # 如果模型的创建时间到现在的时间段数大于模型估计间隔 every_group_len
    # 重新用 train_min_len 段时间估计模型
    if len(df[model_with_time._time :]) > every_group_len:
        model_with_time = create_model_with_time()
        model = model_with_time._model
    # 模型仍在有效期
    else:
        model: hmm.GaussianHMM = model_with_time._model
# 不存在模型则直接估计
else:
    model_with_time = create_model_with_time()
    model = model_with_time._model

# print(model_with_time)


################################
# step 3 信号

# 观测指标的 df
ob_df = df[indicator_list].loc[model_with_time._time :]
start, *_, end = ob_df.index
# print(ob_df)
ob_np = hmmquant.utils.normalization(ob_df, plus=2).values  # type:ignore
_, state = model.decode(ob_np, algorithm="viterbi")
# 只关心当前最后一个隐藏状态，我们用它来预测下一个状态是属于涨组还是跌组
last_state = state[-1]

r = hmmquant.utils.get_all_state_rr(LOGRR[start:end], state)
state_group = hmmquant.model.distinguish_state(r, 1, 1)

if last_state in state_group.rise_state:
    sig = 1
elif last_state in state_group.fall_state:
    sig = -1
else:
    sig = 0

print(df.index[-1], sig, sep=",")
