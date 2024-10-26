import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts

# 设置tushare token
# tushare_token = 'your_tushare_token'
tushare_token = '0858b935f4518d9e56ffeb19910dc13e296291364ea1d7bd574eb84b'
ts.set_token(tushare_token)
pro = ts.pro_api()

# 获取股票数据
df = pro.daily(ts_code='000001.SZ', start_date='20210101', end_date='20230630', fields='trade_date,close')

# 计算短期均线和长期均线
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA20'] = df['close'].rolling(window=20).mean()

# 生成交易信号
df['signal'] = np.where(df['MA5'] > df['MA20'], 1, -1)

# 计算当天持仓状态
df['position'] = df['signal'].diff()

# 回测策略
df['return'] = df['close'].pct_change()  # 计算每日收益率
df['strategy_return'] = df['position'].shift() * df['return']  # 计算每日持仓收益

# 计算累计收益率
df['cumulative_return'] = (1 + df['strategy_return']).cumprod()

# 可视化结果
plt.plot(df['trade_date'], df['cumulative_return'])
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.title('Dual Moving Average Strategy')
plt.xticks(rotation=45)
plt.show()
