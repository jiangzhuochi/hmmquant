from datetime import datetime
from pathlib import Path

import pandas as pd
import talib

from hmmquant import utils

DATA_DIR = Path(".") / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def update_data(update=True) -> pd.DataFrame:

    name = datetime.today().strftime(r"%Y-%m-%d") + ".pkl"
    # 只有文件存在且不更新时才直接返回
    if (DATA_DIR / name).is_file() and not update:
        return pd.read_pickle(DATA_DIR / name)

    data = pd.read_csv(DATA_DIR / "data_wind.csv", index_col=0, parse_dates=True)
    high_se = data["HIGH"]
    low_se = data["LOW"]
    close_se = data["CLOSE"]
    amount_se = data["AMOUNT"]

    LOGRR: pd.Series = utils.get_logrr(close_se)
    MA5: pd.Series = talib.SMA(close_se, timeperiod=5)  # type:ignore
    MA200: pd.Series = talib.SMA(close_se, timeperiod=200)  # type:ignore
    macd, _, _ = talib.MACD(  # type:ignore
        close_se, fastperiod=12, slowperiod=26, signalperiod=9
    )  # type:ignore
    MACD: pd.Series = macd.dropna()
    CCI: pd.Series = talib.CCI(high_se, low_se, close_se, timeperiod=14)  # type:ignore
    RSI: pd.Series = talib.RSI(close_se, timeperiod=14)  # type:ignore

    df = pd.DataFrame(
        {
            "close_se": close_se,
            "LOGRR": LOGRR,
            "MA5": MA5,
            "MA200": MA200,
            "MACD": MACD,
            "CCI": CCI,
            "RSI": RSI,
        }
    ).dropna()
    df.to_pickle(DATA_DIR / name)
    return df


INDICATOR = update_data(update=False)
