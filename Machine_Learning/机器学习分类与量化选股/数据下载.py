import tushare as ts
import pandas as pd
import os
import numpy as np
import time
from tqdm import tqdm

"""
获取历史数据
"""

save_path = './stock'
if not os.path.exists(save_path):
    os.mkdir(save_path)
    
# 存储股票交易信息    
stock_inf_path = os.path.join(base_path,  'OldData')
if not os.path.exists(stock_inf_path):
    os.mkdir(stock_inf_path)
    
company_path = os.path.join(save_path, 'company_info.csv')

# 设置起始日期
startdate = '20180701'
enddate = '20230630'


# 获取基础信息数据，包括股票代码、名称、上市日期、退市日期等    
if os.path.exists(company_path):
    pool = pd.read_csv(company_path, encoding='utf-8', index_col = 0)
else:
    mytoken = ' '
    ts.set_token(mytoken)
    pro = ts.pro_api()
    pool = pro.stock_basic(exchange='',
                           list_status='L',
                           adj='qfq',
                           fields='ts_code, symbol,name,area,industry,fullname,list_date, market,exchange,is_hs')

    # 因为穷没开通创业板和科创板权限，这里只考虑主板和中心板
    pool = pool[pool['market'].isin(['主板', '中小板'])].reset_index()
    pool.to_csv(os.path.join(save_path, 'company_info.csv'), index=False, encoding='utf-8')

print('获得上市股票总数：', len(pool)-1)
