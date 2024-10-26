import numpy as np
import pandas as pd
import statsmodels.api as sm

# 假设我们有一个包含市值因子和收益的数据框 DataFrame
# 数据框的列包括：'日期'、'股票代码'、'市值'、'收益'等

# 假设我们已经从数据源加载了数据，存储在变量 data 中
data = [
    
]
# 选择所需的列
data = data[['日期', '股票代码', '市值', '收益']]

# 根据日期进行分组
groups = data.groupby('日期')

# 定义一个函数来执行市值中性化
def market_neutralize(group):
    # 提取市值和收益的数据列
    market_cap = group['市值']
    returns = group['收益']

    # 添加截距项
    X = sm.add_constant(market_cap)

    # 执行线性回归，拟合收益率与市值的关系
    model = sm.OLS(returns, X)
    results = model.fit()

    # 提取回归系数
    beta = results.params['市值']

    # 计算市值中性化后的收益
    neutralized_returns = returns - beta * market_cap

    # 将市值中性化后的收益添加到数据框中
    group['市值中性化收益'] = neutralized_returns

    return group

# 对每个日期的数据进行市值中性化
neutralized_data = groups.apply(market_neutralize)
