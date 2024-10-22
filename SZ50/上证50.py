import baostock as bs # 导入 baostock 库
import pandas as pd   # 导入 pandas 库
from IPython.display import display

# 登陆系统
lg = bs.login()  # 调用 login 方法进行登陆
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code) # 打印登陆返回错误码
print('login respond  error_msg:'+lg.error_msg)  # 打印登陆返回错误信息

# 获取上证50成分股
rs = bs.query_sz50_stocks()  # 调用 query_sz50_stocks 方法获取上证50成分股信息
print('query_sz50 error_code:'+rs.error_code)  # 打印方法返回错误码
print('query_sz50  error_msg:'+rs.error_msg)   # 打印方法返回错误信息

# 打印结果集
sz50_stocks = []   # 创建一个空列表，用于存储查询结果
while (rs.error_code == '0') & rs.next():
    # 如果查询没有出错且还有数据
    sz50_stocks.append(rs.get_row_data())   # 将获取到的数据添加到列表中
result = pd.DataFrame(sz50_stocks, columns=rs.fields)  # 使用 pandas 将数据转换为 DataFrame 格式
# 结果集输出到csv文件
result.to_csv("./SZ50/sz50_stocks.csv", encoding="gbk", index=False)  # 将结果保存为 csv 文件
display(result)   # 打印结果

# 登出系统
bs.logout()  # 调用 logout 方法进行登出
