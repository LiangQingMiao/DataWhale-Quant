# from bigdatasource.api import DataSource
# from biglearning.api import M
# from biglearning.api import tools as T
# from biglearning.module2.common.data import Outputs
 
# import pandas as pd
# import numpy as np
# import math
# import warnings
# import datetime
 
# from zipline.finance.commission import PerOrder
# from zipline.api import get_open_orders
# from zipline.api import symbol
 
# from bigtrader.sdk import *
# from bigtrader.utils.my_collections import NumPyDeque
# from bigtrader.constant import OrderType
# from bigtrader.constant import Direction

# def m3_initialize_bigquant_run(context):
#     context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))


# def m3_handle_data_bigquant_run(context, data):
#     today = data.current_dt.strftime('%Y-%m-%d')  
#     stock_hold_now = {e.symbol: p.amount * p.last_sale_price
#                  for e, p in context.perf_tracker.position_tracker.positions.items()}

#     cash_for_buy = context.portfolio.cash    
    
#     try:
#         buy_stock = context.daily_stock_buy[today]  
#     except:
#         buy_stock=[]  

#     try:
#         sell_stock = context.daily_stock_sell[today]  
#     except:
#         sell_stock=[] 
    
#     stock_to_sell = [ i for i in stock_hold_now if i in sell_stock ]

#     stock_to_buy = [ i for i in buy_stock if i not in stock_hold_now ]  

#     stock_to_adjust=[ i for i in stock_hold_now if i not in sell_stock ]
    
#     if len(stock_to_sell)>0:
#         for instrument in stock_to_sell:
#             sid = context.symbol(instrument) 
#             cur_position = context.portfolio.positions[sid].amount 
#             if cur_position > 0 and data.can_trade(sid):
#                 context.order_target_percent(sid, 0) 
#                 cash_for_buy += stock_hold_now[instrument]
    

#     if len(stock_to_buy)+len(stock_to_adjust)>0:
#         weight = 1/(len(stock_to_buy)+len(stock_to_adjust)) 
#         for instrument in stock_to_buy+stock_to_adjust:
#             sid = context.symbol(instrument) 
#             if  data.can_trade(sid):
#                 context.order_target_value(sid, weight*cash_for_buy) 

# def m3_prepare_bigquant_run(context):
#     df = context.options['data'].read_df()

#     def open_pos_con(df):
#         return list(df[df['buy_condition']>0].instrument)

#     def close_pos_con(df):
#         return list(df[df['sell_condition']>0].instrument)

#     context.daily_stock_buy= df.groupby('date').apply(open_pos_con)

#     context.daily_stock_sell= df.groupby('date').apply(close_pos_con)

# m1 = M.input_features.v1(
#     features="""# #号开始的表示注释
# # 多个特征，每行一个，可以包含基础特征和衍生特征
# buy_condition=where(mean(close_0,5)>mean(close_0,10),1,0)
# sell_condition=where(mean(close_0,5)<mean(close_0,10),1,0)""",
#     m_cached=False
# )

# m2 = M.instruments.v2(
#     start_date=T.live_run_param('trading_date', '2019-03-01'),
#     end_date=T.live_run_param('trading_date', '2021-06-01'),
#     market='CN_STOCK_A',
#     instrument_list="""600519.SHA
# 601392.SHA""",
#     max_count=0
# )

# m7 = M.general_feature_extractor.v7(
#     instruments=m2.data,
#     features=m1.data,
#     start_date='',
#     end_date='',
#     before_start_days=60
# )

# m8 = M.derived_feature_extractor.v3(
#     input_data=m7.data,
#     features=m1.data,
#     date_col='date',
#     instrument_col='instrument',
#     drop_na=False,
#     remove_extra_columns=False
# )

# m4 = M.dropnan.v2(
#     input_data=m8.data
# )

# m3 = M.trade.v4(
#     instruments=m2.data,
#     options_data=m4.data,
#     start_date='',
#     end_date='',
#     initialize=m3_initialize_bigquant_run,
#     handle_data=m3_handle_data_bigquant_run,
#     prepare=m3_prepare_bigquant_run,
#     volume_limit=0.025,
#     order_price_field_buy='open',
#     order_price_field_sell='open',
#     capital_base=1000000,
#     auto_cancel_non_tradable_orders=True,
#     data_frequency='daily',
#     price_type='后复权',
#     product_type='股票',
#     plot_charts=True,
#     backtest_only=False,
#     benchmark='000300.HIX'
# )
