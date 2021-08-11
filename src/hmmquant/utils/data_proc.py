import numpy as np
import pandas as pd
import requests


# https://github.com/jindaxiang/akshare/blob/a3dc77a5e6801e5ef0ba77ff1bc5edca48968af5/akshare/futures_derivative/sina_futures_index.py#L83
def futures_main_sina(symbol: str = "V0", trade_date: str = "20191225") -> pd.DataFrame:
    """
    新浪财经-期货-主力连续日数据
    http://vip.stock.finance.sina.com.cn/quotes_service/view/qihuohangqing.html#titlePos_1
    :param symbol: 通过 futures_display_main_sina 函数获取 symbol
    :type symbol: str
    :param trade_date: 交易日
    :type trade_date: str
    :return: 主力连续日数据
    :rtype: pandas.DataFrame
    """
    trade_date = trade_date[:4] + "_" + trade_date[4:6] + "_" + trade_date[6:]
    url = f"https://stock2.finance.sina.com.cn/futures/api/jsonp.php/var%20_{symbol}{trade_date}=/InnerFuturesNewService.getDailyKLine?symbol={symbol}&_={trade_date}"
    resp = requests.get(url)
    data_json = resp.text[resp.text.find("([") + 1 : resp.text.rfind("])") + 1]
    data_df: pd.DataFrame = pd.read_json(data_json)  # type: ignore
    data_df.columns = ["date", "o", "h", "l", "c", "成交量", "持仓量"]
    return data_df


