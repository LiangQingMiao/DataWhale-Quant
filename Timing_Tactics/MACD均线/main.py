# import numpy as np
# import pandas as pd
# import talib as ta 
# # Joinquant数据下载API
# from jqdatasdk import *
# # 登录验证
# auth("Your_phonenum", "Your_passwd")
# import matplotlib.pyplot as plt

# # 下载数据
# # 下载2015-2018年的沪深300指数，频率为每天，只要收盘价
# price = get_price("000300.XSHG", start_date="2015-01-01", end_date="2018-12-31", frequency="daily", fields=['close'])['close']
# # 用python自带的tseries库中的pct_change()函数计算日收益率
# ret = price.pct_change()

# # 用talib库中的相应函数计算MACD指标
# dif, dea, macd = ta.MACD(price)
# # 只考虑MACD指标，MACD转正时开仓买入，转负时清仓
# sig = (macd>0)

# # sig滞后一期、去除空值、转换成整数
# sig_lag = sig.shift(1).fillna(0).astype(int)
# # sig_lag与股票日收益率相乘，即可得策略日收益率。python能自动对齐时间序列的日期。
# sig_ret = sig_lag*ret
# # 计算策略累计收益
# cum_sig_ret = (1+sig_ret).cumprod()

# # 把股票价格转换成从1开始，方便比较
# price_norm = price/price[0]

# # 简单起见，这里不考虑手续费，作图比较该策略和“买入-持有”策略的表现。
# plt.figure(figsize=(18,8))
# plt.plot(price_norm)
# plt.plot(cum_sig_ret)
# plt.legend(["benchmark", "strategy cumulative return"], loc="upper left")
# plt.show()
