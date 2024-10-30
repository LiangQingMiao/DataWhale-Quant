import datetime
import tushare as ts

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

ts.set_token(ts_token) 
pro = ts.pro_api()

time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
end_dt = time_temp.strftime('%Y%m%d')

df = ts.pro_bar(ts_code='000001.SZ', start_date='20150101', end_date=end_dt, freq='D')

#把数据按时间调转顺序，最新的放后面，从 tushare 下载的数据是最新的在前面，为了后面准备 X,y 数据方便
df = df.iloc[::-1]
df.reset_index(inplace=True)

df.head() #用 df.head() 可以查看一下下载下来的股票价格数据，显示数据如下：



#只用数据里面的收盘价字段的数据，也可以测试用更多价格字段作为预测输入数据
data_set = df.loc[:, ['close']]
#只取价格数据，不要表头等内容
data_set = data_set.values
# #对数据做规则化处理，都按比例转成 0 到 1 之间的数据，这是为了避免真实数据过大或过小影响模型判断
sc = MinMaxScaler(feature_range = (0, 1))
data_set_scaled = sc.fit_transform(data_set)
 
print(data_set_scaled.shape)
 
plt.figure()
plt.plot(data_set_scaled,"r-")
