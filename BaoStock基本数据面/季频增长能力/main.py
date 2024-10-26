import baostock as bs
import pandas as pd
from IPython.display import display

# 登录 BaoStock 系统
lg = bs.login()

# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 获取600036招商银行季频成长能力数据
growth_list = []
rs_growth = bs.query_growth_data(code="sh.600036", year=2022, quarter=4)
while (rs_growth.error_code == '0') & rs_growth.next():
    growth_list.append(rs_growth.get_row_data())

# 转换为DataFrame格式
df_growth = pd.DataFrame(growth_list, columns=rs_growth.fields)

# 打印输出
display(df_growth)

# 将结果集输出到csv文件
df_growth.to_csv("D:\\growth_data.csv", encoding="gbk", index=False)

# 退出 BaoStock 系统
bs.logout()
