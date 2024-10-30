import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
plt.style.use('seaborn')

plt.figure(figsize=(20,8))
# 下载标普500的数据
data = yf.download('^GSPC', start='2015-01-01', end='2022-12-31')

# 选择收盘价格
close = data['Close']

# 进行季节性分解
result = seasonal_decompose(close, model='additive', period=252)

# 绘制趋势，季节性和残差
result.plot()
plt.show()
