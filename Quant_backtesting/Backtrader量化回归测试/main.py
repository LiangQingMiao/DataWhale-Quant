# 导入函数库
from __future__ import (absolute_import, division, print_function, unicode_literals) 
import datetime
import pymysql
import pandas as pd
import backtrader as bt
import tushare as ts
import numpy as np


# 数据获取(从Tushare中获取数据)
""" 
数据获取一般都是通过连接数据库从数据库中读取，对于不了解数据库来源的新手可以从Tushare中直接获取数据
"""
def get_data(stock_code):
    """
    stock_code:股票代码,类型: str
    return: 股票日线数据,类型: DataFrame
    """
    token = 'Tushare token'   # 可通过进入个人主页-接口TOKEN获得

    ts.set_token(token)
    pro = ts.pro_api(token)

    data_daily = pro.daily(ts_code = stock_code, start_date='20180101', end_date='20230101')
    data_daily['trade_date'] = pd.to_datetime(data_daily['trade_date'])
    data_daily = data_daily.rename(columns={'vol': 'volume'})
    data_daily.set_index('trade_date', inplace=True) 
    data_daily = data_daily.sort_index(ascending=True)
    dataframe = data_daily
    data_daily['openinterest'] = 0
    dataframe['openinterest'] = 0
    data = bt.feeds.PandasData(dataname=dataframe,
                               fromdate=datetime.datetime(2018, 1, 1),
                               todate=datetime.datetime(2023, 1, 1)
                               )

    return data


# 双均线策略实现
class DoubleAverages(bt.Strategy):

    # 设置均线周期
    params = (
        ('period_data5', 5),
        ('period_data10', 10)
    )

    # 日志输出
    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # 初始化数据参数

        self.dataclose = self.datas[0].close   # 定义变量dataclose,保存收盘价
        self.order = None   # 定义变量order,用于保存订单
        self.buycomm = None    # 定义变量buycomm,记录订单佣金
        self.buyprice = None    # 定义变量buyprice,记录订单价格

        self.sma5 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.period_data5)  # 计算5日均线
        self.sma10 = bt.indicators.SimpleMovingAverage(self.datas[0], period=self.params.period_data10)  # 计算10日均线

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:  # 若订单提交或者已经接受则返回
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'Buy Executed, Price: %.2f, Cost: %.2f, Comm: %.2f' %
                    (order.executed.price, order.executed.value, order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:
                self.log('Sell Executed, Price: %.2f, Cost: %.2f, Comm: %.2f' %
                        (order.executed.price, order.executed.value, order.executed.comm))
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None


    def notify_trade(self, trade):
        if not trade.isclosed:   # 若交易未关闭则返回
            return
        self.log('Operation Profit, Total_Profit %.2f, Net_Profit: %.2f' %
                (trade.pnl, trade.pnlcomm))    # pnl表示盈利, pnlcomm表示手续费

    def next(self):   # 双均线策略逻辑实现
        self.log('Close: %.2f' % self.dataclose[0])   # 打印收盘价格

        if self.order:   # 检查是否有订单发送
            return

        if not self.position:   # 检查是否有仓位
            if self.sma5[0] > self.sma10[0]:
                self.log('Buy: %.2f' % self.dataclose[0])
                self.order = self.buy()

        else:
            if self.sma5[0] < self.sma10[0]:
                self.log('Sell: %.2f' % self.dataclose[0])
                self.order = self.sell()





if __name__ == '__main__':
    cerebro = bt.Cerebro()   # 创建策略容器
    cerebro.addstrategy(DoubleAverages)    # 添加双均线策略
    data = get_data('000001.SZ')
    cerebro.adddata(data)   # 添加数据
    cerebro.broker.setcash(10000.0)   # 设置资金
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)   # 设置每笔交易的股票数量
    cerebro.broker.setcommission(commission=0.01)   # 设置手续费
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())   # 打印初始资金
    cerebro.run()   # 运行策略
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())   # 打印最终资金
    cerebro.plot()
