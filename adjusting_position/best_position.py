# import akshare as ak
# import datetime
# import warnings
# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize
# warnings.filterwarnings('ignore')

# def get_ret(code,start_date,end_date):
#     data = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start_date, end_date=end_date, adjust="hfq")
#     data.index = pd.to_datetime(data['日期'], format='%Y-%m-%d')  # 设置日期索引
#     close = data['收盘']  # 日收盘价
#     close.name = code
#     ret = close.pct_change() # 日收益率
#     return ret

# end_date = datetime.datetime.now().strftime('%Y%m%d')
# index_stock_cons_weight_csindex_df = ak.index_stock_cons_weight_csindex(symbol="000016")
# stock_codes = index_stock_cons_weight_csindex_df['成分券代码'].to_list()
# start_date =(index_stock_cons_weight_csindex_df['日期'].iat[0] - pd.Timedelta(days=365*1)).strftime('%Y%m%d')

# ret_list = []
# for code in stock_codes:
#     ret = get_ret(code,start_date=start_date,end_date=end_date)
#     ret_list.append(ret)
# df_ret = pd.concat(ret_list,axis=1).dropna()
# # 数据和模型都准备好后，我们就可以在时间轴上滚动计算最优化模型的权重，注意不要使用未来数据。
# records = []
# for trade_date in df_ret.loc[index_stock_cons_weight_csindex_df['日期'].iat[0]:].index.to_list():
#     df_train = df_ret.loc[:trade_date].iloc[-1-240:-1]
#     df_test = df_ret.loc[trade_date]
#     StockSharpDf = get_weights(df_train, target='sharp')  # 最大夏普组合
#     StockRPDf = get_weights(df_train, target='rp')  # 风险平价组合
#     StockVarDf = get_weights(df_train, target='var')  # 最小风险组合
#     records.append([trade_date,
#                     (df_test.mul(StockSharpDf)).sum(),
#                     (df_test.mul(StockRPDf)).sum(),
#                     (df_test.mul(StockVarDf)).sum(),
#                     df_test.mean()])
# df_record = pd.DataFrame(records,columns=['日期','最大夏普组合','风险平价组合','最小风险组合','等权重组合'])
# df_record = df_record.set_index('日期')

# # 定义一些辅助函数
# def get_weights(df: pd.DataFrame, target='sharp', canshort=False) -> pd.Series:
#     '''
#     :param df: 资产日度涨跌幅矩阵
#     :param target: 优化目标 sharp→最大夏普比组合 rp→风险平价组合  var→最小风险组合
#     :param canshort: 是否可以做空
#     :return: 组合比率
#     '''
#     MeanReturn = df.mean().values  # 期望收益
#     Cov = df.cov()  # 协方差
   
# # 定义优化函数、初始值、约束条件
# # 负夏普比
# def neg_sharp(w):
#     R = w @ MeanReturn
#     Var = w @ Cov @ w.T
#     sharp = R / Var ** 0.5
#     return -sharp * np.sqrt(240)

# # 风险
# def variance(w):
#     Var = w @ Cov @ w.T
#     return Var * 10000

# def RiskParity(w):
#     weights = np.array(w)  # weights为一维数组
#     sigma = np.sqrt(np.dot(weights, np.dot(Cov, weights)))  # 获取组合标准差
#     # sigma = np.sqrt(weights@cov@weights)
#     MRC = np.dot(Cov, weights) / sigma  # MRC = Cov@weights/sigma
#     # MRC = np.dot(weights,cov)/sigma
#     TRC = weights * MRC
#     delta_TRC = [sum((i - TRC) ** 2) for i in TRC]
#     return sum(delta_TRC)

# # 设置初始值
# w0 = np.ones(df.shape[1]) / df.shape[1]
# # 约束条件 w之和为1
# cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
# bnds = tuple((0, 1) for w in w0)  # 做多的约束条件，如果可以做空，则不传入该条件

# if target == 'sharp':
#     fc = neg_sharp
# elif target == 'rp':
#     fc = RiskParity
# elif target == 'var':
#     fc = variance

# if canshort:
#     res = minimize(fc, w0, method='SLSQP', constraints=cons, options={'maxiter': 100})
# else:
#     res = minimize(fc, w0, method='SLSQP', constraints=cons, options={'maxiter': 100}, bounds=bnds)

# # if target == 'sharp':
# #     print('最高夏普:', -res.fun)
# # elif target == 'rp':
# #     print('风险平价:', res.fun)
# # elif target == 'var':
# #     print('最低风险:', res.fun)

# # print('最优比率:', res.x)
# # print('年化收益:', ReturnYearly(res.x) * 100, "%")
# weight = pd.Series(res.x, index=df.columns)
# return weight
