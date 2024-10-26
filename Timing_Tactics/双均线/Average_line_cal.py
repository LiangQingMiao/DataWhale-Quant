"""
使用tushare库来获取股票数据，并对数据进行处理和计算移动平均值。
"""

import tushare as ts

# https://tushare.pro/user/token
tushare_token = '0858b935f4518d9e56ffeb19910dc13e296291364ea1d7bd574eb84b'
ts.set_token(tushare_token)
pro = ts.pro_api(tushare_token)

# 个股数据 https://tushare.pro/document/2?doc_id=27
# 从tushare获取个股数据，从2023年5月1日到2023年6月30日期间获取股票代码为000001.SZ的数据，字段包括交易代码、交易日期和收盘价。
df = pro.daily(ts_code='000001.SZ', start_date='20230501', end_date='20230630', fields='ts_code,trade_date,close')
df = df.sort_values(by='trade_date')

# 使用.rolling()函数计算收盘价的移动平均值，分别为5天、10天、20天和30天，并将结果存储在M5、M10、M20和M30列中。
df['M5'] = df.close.rolling(window=5).mean().round(2)
df['M10'] = df.close.rolling(window=10).mean().round(2)
df['M20'] = df.close.rolling(window=20).mean().round(2)
df['M30'] = df.close.rolling(window=30).mean().round(2)

# 对数据按交易日期进行降序排序，以便按照日期从新到旧的顺序显示数据。
df = df.sort_values(by='trade_date', ascending=False)
print("前10条数据：\n", df.head(10))
df.to_csv('result.csv', index=False)
