from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import os.path
import sys

import backtrader as bt
import backtrader.feeds as btfeeds
import pandas as pd

sig_df = pd.read_csv(
    "./csv/RSI/sig,4,170,340.csv", index_col=0, parse_dates=["quarter_time"]
)

# print(sig_df.index)
# print(sig_df.loc['2015-08-03 11:15'][0])
# backtrader 简直是 linter 杀手233
class MyHLOC(btfeeds.GenericCSVData):  # type: ignore

    lines = ("sig",)

    params = (
        ("nullvalue", 0.0),
        # ("dtformat", (r"%Y-%m-%d %H:%M:%S")),
        ("dtformat", (r"%Y/%m/%d %H:%M")),
        ("datetime", 0),
        ("time", -1),
        ("high", 1),
        ("low", 2),
        ("open", 3),
        ("close", 4),
        ("volume", 5),
        ("openinterest", -1),
        ("sig", 11),
    )


class MySizer(bt.Sizer):
    """其实由于价格波动很小，
    要么多 50 手，要么空 50 手，要么 0 手"""

    def _getsizing(self, comminfo, cash, data, isbuy):

        position = self.broker.getposition(data)  # type: ignore

        # 买入时
        if isbuy:
            # 开多仓，把所有的钱花出去
            # 因为是以下一个价格买入的
            # 少买一手防止价格变高导致钱不够
            if position.size == 0:
                return cash // data.close[0] - 1
            # 如果有多仓，则不操作
            elif position.size > 0:
                return 0
            # 当空仓时，反向开仓
            else:
                return (-position.size) * 2

        # 卖出时
        else:
            # 如果没有仓位，空仓限制为多仓的手数
            if position.size == 0:
                return cash // data.close[0] - 1
            elif position.size > 0:
                return position.size * 2
            else:
                return 0


# Create a Stratey
class TestStrategy(bt.Strategy):
    def log(self, txt, dt=None):
        """Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.datetime(0)
        print("%s, %s" % (dt.strftime(r"%Y-%m-%d %H:%M:%S"), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.datasig = self.datas[0].sig

    def next(self):
        print(self.position.size)
        c_time = self.data.datetime.datetime(0).strftime(r"%Y-%m-%d %H:%M:%S")
        try:
            c_sig = sig_df.loc[c_time][0]
        except KeyError:
            c_sig = 0
        if c_sig == 1:
            self.buy()
        elif c_sig == -1:
            self.sell()
        # Simply log the closing price of the series from the reference
        self.log("Close, %.4f" % self.dataclose[0])


if __name__ == "__main__":

    # Create a cerebro entity
    cerebro = bt.Cerebro()
    # analyzer

    cerebro.addstrategy(TestStrategy)

    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, "../../data/quarter.csv")

    # 注意 timeframe 设置成分钟级
    data = MyHLOC(
        dataname=datapath,
        timeframe=bt.TimeFrame.Minutes,  # type: ignore
    )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(5000.0)

    # Print out the starting conditions
    print("Starting Portfolio Value: %.2f" % cerebro.broker.getvalue())

    cerebro.addsizer(MySizer)

    cerebro.addanalyzer(
        bt.analyzers.Returns,
        _name="datareturns",
        tann=17 * 250,
    )

    cerebro.addanalyzer(
        bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Years, _name="timereturns"  # type: ignore
    )

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="ana")

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=0, factor=17 * 250)

    # Run over everything
    results = cerebro.run()

    # Print out the final result
    print("Final Portfolio Value: %.2f" % cerebro.broker.getvalue())
    # cerebro.plot()

    strat0 = results[0]

    tret_analyzer = strat0.analyzers.getbyname("timereturns")
    print(tret_analyzer.get_analysis())
    tdata_analyzer = strat0.analyzers.getbyname("datareturns")
    print(tdata_analyzer.get_analysis())
    tdata_analyzer = strat0.analyzers.getbyname("ana")
    print(tdata_analyzer.get_analysis())
    tdata_analyzer = strat0.analyzers.getbyname("sharperatio")
    print(tdata_analyzer.get_analysis())
