import json
import pickle
import re
import sys
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Union

from loguru import logger as _logger
from scipy import stats

if TYPE_CHECKING:
    from loguru import Logger

import numpy as np
import pandas as pd
import requests
import talib
from apscheduler.schedulers.blocking import BlockingScheduler
from hmmlearn import hmm

StateGroup = namedtuple("StateGroup", ["rise_state", "fall_state", "shock_state"])


def normalization(raw_data: Union[pd.Series, pd.DataFrame], plus=2) -> pd.DataFrame:
    def _norm(se: pd.Series):
        # z_score标准化
        mean, std = se.describe()[["mean", "std"]]
        z_score_scaling = (se - mean) / std

        # minmax标准化
        ma, mi = z_score_scaling.describe()[["max", "min"]]
        min_max_scaling = (z_score_scaling - mi) / (ma - mi) + plus

        # 使用boxcox
        boxcoxed_data, _ = stats.boxcox(min_max_scaling)  # type: ignore
        return pd.Series(boxcoxed_data, index=se.index)

    # print(raw_data)
    raw_data = pd.DataFrame(raw_data)
    # print(raw_data)
    if isinstance(raw_data, pd.Series):
        return pd.DataFrame(_norm(raw_data))
    else:
        return raw_data.apply(_norm)


def get_logrr(close: Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]:
    logrr = np.log(close).diff()[1:]  # type: ignore
    return logrr


def get_state_rr(rr_seq: pd.Series, state_seq: np.ndarray, target_s) -> pd.Series:
    """得到某个状态的收益"""
    rrlist = []
    for r, s in zip(rr_seq.values, state_seq):
        if s == target_s:
            rrlist.append(r)
        else:
            rrlist.append(0)
    return pd.Series(rrlist, index=rr_seq.index)


def get_all_state_rr(rr_seq: pd.Series, state_seq: np.ndarray) -> pd.DataFrame:
    """得到所有状态的收益"""
    _d = {}
    all_state = sorted(list(set(state_seq)))
    for s in all_state:
        _d[s] = get_state_rr(rr_seq, state_seq, s)
    return pd.DataFrame(_d)


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


########## log
LOG_PATH = Path(".") / "logs"

LOG_PATH.mkdir(parents=True, exist_ok=True)
LOG_FORMAT = (
    "<level><v>{level:<8}</v>"
    " [{time:YYYY-MM-DD} {time:HH:mm:ss.SSS} <d>{module}:{name}:{line}</d>]</level>"
    " {message}"
)
LOG_LEVEL = "TRACE"


def make_filter(name):
    def filter(record):
        return record["extra"].get("name") == name

    return filter


logger: "Logger" = _logger.opt(colors=True)
logger.remove()


logger.add(
    LOG_PATH / "signal.log",
    format="{message}",
    level=LOG_LEVEL,
    encoding="utf-8",
    filter=make_filter("signal"),
)
logger.add(
    LOG_PATH / "rr.log",
    format="{message}",
    level=LOG_LEVEL,
    encoding="utf-8",
    filter=make_filter("rr"),
)
logger.add(
    sys.stdout,
    format="{message}",
    level=LOG_LEVEL,
    filter=make_filter("stdout"),
)

# 记录信号
signal_logger = logger.bind(name="signal")
# 记录收益
rr_logger = logger.bind(name="rr")
# 标准输出
stdout_logger = logger.bind(name="stdout")
########


DATA_DIR = Path(".") / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
sched = BlockingScheduler()


def get_data() -> pd.DataFrame:
    """从网上爬数据, 转换成 DataFrame"""

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
                **{
                    col_name: "float64" for col_name in ["open", "high", "low", "close"]
                },
                **{col_name: "int64" for col_name in ["volume", "position"]},
            }
        )
    )
    # 如果现在的时间在 df 最后一行的时间之前
    # 就抛弃最后一行
    if (datetime.now() - df.index[-1]).total_seconds() < 0:  # type:ignore
        df = df.head(-1)
    return df


################################
# step 1 读取数据计算指标

# quarter_data = pd.read_csv(DATA_DIR / "quarter.csv", index_col=0, parse_dates=True)
def calc_indicator(quarter_data: pd.DataFrame):

    close_se = quarter_data["close"]
    # print(close_se)
    LOGRR: pd.Series = get_logrr(close_se)  # type:ignore
    RSI: pd.Series = talib.RSI(close_se, timeperiod=14)  # type:ignore
    return pd.DataFrame(
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


def create_model_with_time(only_indicator_df: pd.DataFrame):

    # 取 train_min_len 个估计模型
    needed_df = only_indicator_df.tail(train_min_len)
    start, *_ = needed_df.index

    train = needed_df
    train_np = normalization(train, plus=2).values  # type:ignore
    model = run_model(train_np, state_num)

    # 测试用
    # start = datetime.strptime("2021-07-20 15:15", r"%Y-%m-%d %H:%M")
    ############

    model_with_time = ModelWithTime(
        _model=model, _time=start.strftime(r"%Y-%m-%d %H:%M")  # type:ignore
    )
    # 保存模型
    with open(DATA_DIR / "model.pkl", "wb") as f:
        pickle.dump(model_with_time, f)
    return model_with_time


def get_model(only_indicator_df: pd.DataFrame):

    if (model_path := DATA_DIR / "model.pkl").exists():
        with open(model_path, "rb") as f:
            model_with_time = pickle.load(f)

        # 如果模型的创建时间到现在的时间段数大于模型估计间隔 every_group_len
        # 重新用 train_min_len 段时间估计模型
        if len(only_indicator_df.loc[model_with_time._time :]) > every_group_len:
            model_with_time = create_model_with_time(only_indicator_df)
            model = model_with_time._model
            # print("估计")
            # print(model_with_time)
        # 模型仍在有效期
        else:
            model: hmm.GaussianHMM = model_with_time._model
    # 不存在模型则直接估计
    else:
        model_with_time = create_model_with_time(only_indicator_df)
        model = model_with_time._model

    return model, model_with_time._time


################################
# step 3 信号
def emit_signal(
    model: hmm.GaussianHMM, model_estimation_time: str, all_data: pd.DataFrame
):
    # 观测序列
    ob_df = all_data[indicator_list].loc[model_estimation_time:]
    start, *_, end = ob_df.index
    # print(ob_df)
    ob_np = normalization(ob_df, plus=2).values  # type:ignore
    _, state = model.decode(ob_np, algorithm="viterbi")
    # 只关心当前最后一个隐藏状态，我们用它来预测下一个状态是属于涨组还是跌组
    last_state = state[-1]

    r = get_all_state_rr(all_data["LOGRR"][start:end], state)
    state_group = distinguish_state(r, 1, 1)

    if last_state in state_group.rise_state:
        sig = 1
    elif last_state in state_group.fall_state:
        sig = -1
    else:
        sig = 0
    return sig


################################
# step 0 参数


@sched.scheduled_job(
    "cron",
    id="my_job_id",
    day_of_week="mon-fri",
    hour="9-11,13-15",
    minute="0,15,30,45",
    second="1",
)
def job_function():

    now = datetime.now()

    if (
        (now.hour == 9 and now.minute in [0, 15, 30])
        or (now.hour == 11 and now.minute in [45])
        or (now.hour == 13 and now.minute in [0])
        or (now.hour == 15 and now.minute in [30, 45])
    ):
        return

    all_data = calc_indicator(get_data())

    only_indicator_df = all_data[indicator_list]
    model, model_time = get_model(only_indicator_df)
    # print(model, model_time)
    sig = emit_signal(model, model_time, all_data)
    try:
        with open(DATA_DIR / "last_sig.txt", "r", encoding="utf8") as f:
            try:
                last_sig = int(f.read())
            except ValueError:
                last_sig = 0
    except FileNotFoundError:
        last_sig = 0
    with open(DATA_DIR / "last_sig.txt", "w", encoding="utf8") as f:
        f.write(str(sig))
    stdout_logger.info(f"{all_data.index[-1]},{sig}")
    signal_logger.info(f"{all_data.index[-1]},{sig}")

    # 纯多在上个时段收益
    long_rr = float(all_data["LOGRR"].to_list()[-1])
    # 策略在上个时段的收益
    if last_sig == 0:
        strategy_rr = 0
    elif last_sig == 1:
        strategy_rr = long_rr
    else:
        strategy_rr = -long_rr

    rr_logger.info(f"{all_data.index[-1]},{strategy_rr},{long_rr}")


state_num = 4
train_min_len = 170
every_group_len = 340
indicator_list = ["RSI"]


sched.start()
# job_function()
