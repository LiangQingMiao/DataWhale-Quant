import baostock as bs
import pandas as pd
from IPython.display import display

# 登录 BaoStock 系统
lg = bs.login()

# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 获取600036招商银行季频偿债能力数据
balance_list = []
rs_balance = bs.query_balance_data(code="sh.600036", year=2022, quarter=4)
while (rs_balance.error_code == '0') & rs_balance.next():
    balance_list.append(rs_balance.get_row_data())

# 转换为DataFrame格式
df_balance = pd.DataFrame(balance_list, columns=rs_balance.fields)

# 打印输出
display(df_balance)

# 将结果集输出到csv文件
df_balance.to_csv("D:\\balance_data.csv", encoding="gbk", index=False)

# 退出 BaoStock 系统
bs.logout()
