import baostock as bs
import pandas as pd
from IPython.display import display

# 登录 BaoStock 系统
lg = bs.login()

# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

# 获取600036招商银行季频营运能力数据
operation_list = []
rs_operation = bs.query_operation_data(code="sh.600036", year=2022, quarter=4)
while (rs_operation.error_code == '0') & rs_operation.next():
    operation_list.append(rs_operation.get_row_data())

# 转换为DataFrame格式
df_operation = pd.DataFrame(operation_list, columns=rs_operation.fields)

# 打印输出
display(df_operation)

# 将结果集输出到csv文件
df_operation.to_csv("D:\\operation_data.csv", encoding="gbk", index=False)

# 退出 BaoStock 系统
bs.logout()
