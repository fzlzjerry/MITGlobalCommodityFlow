import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import os
import joblib

# 加载数据以获取港口和商品类型
data_file_path = 'Port level Imports.csv'
data = pd.read_csv(data_file_path)
ports = data['Port'].unique().tolist()
commodities = data['Commodity'].unique().tolist()

# 确保港口和商品类型的字符串格式正确
ports = [port.strip() for port in ports]
commodities = [commodity.strip() for commodity in commodities]

# 创建GUI窗口
root = tk.Tk()
root.title("预测展示")

# 创建下拉框选择港口
port_label = ttk.Label(root, text="选择港口:")
port_label.grid(row=0, column=0, padx=10, pady=10)
port_combo = ttk.Combobox(root, values=ports)
port_combo.grid(row=0, column=1, padx=10, pady=10)

# 创建下拉框选择商品类型
commodity_label = ttk.Label(root, text="选择商品类型:")
commodity_label.grid(row=1, column=0, padx=10, pady=10)
commodity_combo = ttk.Combobox(root, values=commodities)
commodity_combo.grid(row=1, column=1, padx=10, pady=10)

# 创建输入框选择预测时间（月数）
month_label = ttk.Label(root, text="选择预测时间 (月):")
month_label.grid(row=2, column=0, padx=10, pady=10)
month_entry = ttk.Entry(root)
month_entry.grid(row=2, column=1, padx=10, pady=10)

# 创建结果展示标签
result_label = ttk.Label(root, text="预测结果 (kg):")
result_label.grid(row=3, column=0, padx=10, pady=10)
result_value = ttk.Label(root, text="")
result_value.grid(row=3, column=1, padx=10, pady=10)


# 定义查询函数
def query_predictions():
    port = port_combo.get()
    commodity = commodity_combo.get()
    months = int(month_entry.get())

    if port and commodity and months:
        # 加载模型和归一化参数
        model_path = f'models/{port}_{commodity}.keras'
        scaler_path = f'models/{port}_{commodity}_scaler.pkl'

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            result_value.config(text="模型或归一化参数文件不存在")
            return

        model = tf.keras.models.load_model(model_path)
        scaler = joblib.load(scaler_path)

        # 筛选选定港口和商品的数据
        subset = data[(data['Port'] == port) & (data['Commodity'] == commodity)].copy()

        # 去除重量列中的逗号并转换为浮点数
        subset['Containerized Vessel SWT (Gen) (kg)'] = subset['Containerized Vessel SWT (Gen) (kg)'].str.replace(',', '').astype(float)

        # 规范化数据
        subset_scaled = scaler.transform(subset[['Containerized Vessel SWT (Gen) (kg)']])

        # 创建序列
        seq_length = 12
        X = []
        for i in range(len(subset_scaled) - seq_length):
            X.append(subset_scaled[i:(i + seq_length)])
        X = np.array(X)

        if len(X) == 0:
            result_value.config(text="数据不足，无法进行预测")
            return

        last_sequence = X[-1].reshape(1, seq_length, 1)

        # 预测接下来的months个月
        future_predictions = []
        current_sequence = last_sequence

        for _ in range(months):
            next_value = model.predict(current_sequence)
            future_predictions.append(next_value[0, 0])
            next_value_reshaped = np.array([[next_value[0, 0]]]).reshape(1, 1, 1)
            current_sequence = np.append(current_sequence[:, 1:, :], next_value_reshaped, axis=1)

        # 反向转换预测值
        future_predictions_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # 显示预测结果
        result_value.config(text=f"{future_predictions_inv.flatten()} kg")


# 创建查询按钮
query_button = ttk.Button(root, text="查询", command=query_predictions)
query_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

# 运行主循环
root.mainloop()