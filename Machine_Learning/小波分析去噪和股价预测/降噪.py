import pywt 
import numpy as np

#小波去噪处理
data_set_scaled=data_set_scaled.reshape(-1)
 
w = pywt.Wavelet('db8')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(data_set_scaled), w.dec_len)
threshold = 0.05  # Threshold for filtering
 
coeffs = pywt.wavedec(data_set_scaled, 'db8', level=maxlev)  # 将信号进行小波分解
print(coeffs[0].shape)
print(len(coeffs))
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波
    
data_set_scaled_wv = pywt.waverec(coeffs, 'db8')  # 将信号进行小波重构
 
plt.plot(data_set_scaled_wv,"b--")
 
data_set_scaled_wv = np.array(data_set_scaled_wv)
# training_set_scaled=training_set_scaled.reshape(-1,1)
print(data_set_scaled_wv.shape)
