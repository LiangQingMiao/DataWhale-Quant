import baostock as bs
import pandas as pd
from IPython.display import display

# 登录 BaoStock 系统
lg = bs.login()

# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 获取600036招商银行季频盈利能力数据
profit_list = []
rs_profit = bs.query_profit_data(code="sh.600036", year=2022, quarter=4)
while (rs_profit.error_code == '0') & rs_profit.next():
    profit_list.append(rs_profit.get_row_data())

# 转换为DataFrame格式
df_profit = pd.DataFrame(profit_list, columns=rs_profit.fields)

# 打印结果
display(df_profit)

# 将结果集输出到csv文件
df_profit.to_csv("D:\\profit_data.csv", encoding="gbk", index=False)

# 退出 BaoStock 系统
bs.logout()
