import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 读取和准备数据
df = pd.read_csv('stock_data.csv')
X = df[['PE', 'PB', 'ROE']] # 特征因子
y = df['Returns'] # 目标变量

# 拆分训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 构建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 查看模型系数,确定因子权重
print('Factor weights:', model.coef_)

# 使用模型预测测试数据的收益
y_pred = model.predict(X_test)

# 创建一个DataFrame来存储股票的预测收益
predicted_returns = pd.DataFrame({
    'Stock': X_test.index,
    'Predicted return': y_pred
})

# 根据预测的收益选择股票
selected_stocks = predicted_returns[predicted_returns['Predicted return'] > 0.1]

print('Selected stocks:', selected_stocks)
