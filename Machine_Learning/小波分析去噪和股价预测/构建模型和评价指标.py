import pywt 
import numpy as np
import datetime
import tushare as ts
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import LinearSVR
from sklearn.pipeline import make_pipeline
import pandas as pd


import warnings 
warnings.filterwarnings('ignore')


# Calculating directional accuracy
def direct_accuracy(real_Y, pred_Y):
    # Turn numpy into pandas to difference the series
    temp_real = pd.DataFrame(real_Y)
    
    pd_real = temp_real.diff().dropna() # Difference
    
    temp_pred = pred_Y[1:]-real_Y[:-1]
    pd_pred = pd.DataFrame(temp_pred)
    
    # Set the value to 1 if the changes of series is positive
    
    real_direct = np.ones(len(pd_real)) # Default value is set to 1
    pred_direct = np.ones(len(pd_pred))
    
    # Change the value to -1 if the changes is negative
    row, col = np.where(pd_real<0)
    real_direct[row] = -1 
    
    row, col = np.where(pd_pred<0)
    pred_direct[row] = -1
    
    return accuracy_score(real_direct, pred_direct)

def cal_metrics(pred, real, mean_name):
    metric = np.array(['Means', 'MSE', 'MAE'])
    mse = mean_squared_error(real, pred).round(4)
    mae = mean_absolute_error(real, pred).round(4)
    metric = np.vstack((metric, [mean_name, mse, mae]))
    return metric


# ts_token = ''
ts.set_token(ts_token) #需要在 tushare 官网申请一个账号，然后得到 token 后才能通过数据接口获取数据
pro = ts.pro_api()

time_temp = datetime.datetime.now() - datetime.timedelta(days=1)
end_dt = time_temp.strftime('%Y%m%d')

df = ts.pro_bar(ts_code='000001.SZ', start_date='20150101', end_date=end_dt, freq='D')

#把数据按时间调转顺序，最新的放后面，从 tushare 下载的数据是最新的在前面，为了后面准备 X,y 数据方便
df = df.iloc[::-1]
df.reset_index(inplace=True)

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#只用数据里面的收盘价字段的数据，也可以测试用更多价格字段作为预测输入数据
data_set = df.loc[:, ['close']]
#只取价格数据，不要表头等内容
data_set = data_set.values
# #对数据做规则化处理，都按比例转成 0 到 1 之间的数据，这是为了避免真实数据过大或过小影响模型判断
sc = MinMaxScaler(feature_range = (0, 1))
data_set_scaled = sc.fit_transform(data_set)


#小波去噪处理
data_set_scaled=data_set_scaled.reshape(-1)
 
w = pywt.Wavelet('db8')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(data_set_scaled), w.dec_len)
threshold = 0.05  # Threshold for filtering
 
coeffs = pywt.wavedec(data_set_scaled, 'db8', level=maxlev)  # 将信号进行小波分解
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波
    
data_set_scaled_wv = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
 
data_set_scaled_wv = np.array(data_set_scaled_wv)

X_data = []
y_data = []
lags = 30

for i in range(lags, len(data_set_scaled)):
    X_data.append(data_set_scaled_wv[i-lags:i])
    y_data.append(data_set_scaled[i])

X_data, y_data = np.array(X_data), np.array(y_data)

split_date = '20211231'
train_length = df[df['trade_date']== split_date].index.values - lags
X_train = X_data[:train_length[0], :]
y_train = y_data[:train_length[0]]
X_test = X_data[train_length[0]:, :]
y_test = y_data[train_length[0]:]


## 构建线性支持向量机模型
regr = make_pipeline(LinearSVR(random_state=2023, max_iter=1000))

regr.fit(X_train, y_train)

pred_train=regr.predict(X_train)
pred_test=regr.predict(X_test)

metric_train = cal_metrics(pred_train, y_train, 'Train_set')
metric_test = cal_metrics(pred_test, y_test, 'Test_set')
print(metric_train, metric_test)

price_pred_test = sc.inverse_transform(pred_test.reshape(-1, 1))
price_real_test = sc.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(8, 4))
plt.plot(price_pred_test,"b-")
plt.plot(price_real_test,"g-")
 
plt.show()
