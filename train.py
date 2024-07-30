import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Conv1D, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import keras_tuner as kt
import os

# 设置序列长度
seq_length = 12  # 示例序列长度

# 加载数据
file_path = 'Port level Imports.csv'
data = pd.read_csv(file_path)

# 将时间列转换为日期时间格式
data['Time'] = pd.to_datetime(data['Time'])

# 按港口、商品和时间排序数据
data = data.sort_values(by=['Port', 'Commodity', 'Time'])

# 将重量和价值列转换为数值
data['Containerized Vessel SWT (Gen) (kg)'] = data['Containerized Vessel SWT (Gen) (kg)'].str.replace(',', '').astype(
    float)
data['Customs Containerized Vessel Value (Gen) ($US)'] = data[
    'Customs Containerized Vessel Value (Gen) ($US)'].str.replace(',', '').astype(float)


# 定义创建序列的函数
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    if not xs or not ys:
        raise ValueError("No sequences were created. Check the length of the data and seq_length.")
    return np.array(xs), np.array(ys)


# 定义混合模型构建函数
def build_model(hp):
    model = Sequential()
    model.add(Conv1D(filters=hp.Int('filters', min_value=32, max_value=128, step=32),
                     kernel_size=hp.Int('kernel_size', min_value=2, max_value=5, step=1),
                     activation='relu', input_shape=(seq_length, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_conv', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('units_lstm1', min_value=128, max_value=512, step=64),
                   return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_lstm1', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(GRU(units=hp.Int('units_gru1', min_value=128, max_value=512, step=64),
                  return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_gru1', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('units_lstm2', min_value=64, max_value=256, step=64),
                   return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_lstm2', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(GRU(units=hp.Int('units_gru2', min_value=64, max_value=256, step=64),
                  return_sequences=False, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=hp.Float('dropout_gru2', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dense(units=hp.Int('units_dense', min_value=64, max_value=256, step=64), activation='relu',
                    kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
    model.add(Dropout(rate=hp.Float('dropout_dense', min_value=0.2, max_value=0.5, step=0.1)))

    model.add(Dense(1))

    model.compile(optimizer=hp.Choice('optimizer', ['adam', 'adamax', 'nadam']), loss='mse')
    return model


# 设置超参数调整器
tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=100,  # 增加最大epoch数
    factor=2,  # 调整factor以增加初始和后续试验的epoch数
    directory='my_dir',
    project_name='commodity_forecasting'
)

# 参数
batch_size = 256

# 获取唯一的港口和商品
ports = data['Port'].unique()
commodities = data['Commodity'].unique()

# 设置多GPU训练
strategy = tf.distribute.MirroredStrategy()

# 早停、学习率减少和模型检查点的回调函数
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
]

# 创建存储模型的目录
if not os.path.exists('models'):
    os.makedirs('models')

# 遍历每个港口和商品组合
for port in ports:
    for commodity in commodities:
        # 筛选选定港口和商品的数据
        subset = data[(data['Port'] == port) & (data['Commodity'] == commodity)]

        # 检查样本数量是否足够
        if len(subset) < seq_length:
            print(f"Skipping {port} - {commodity} due to insufficient data")
            continue  # 如果数据点不够，则跳过

        # 规范化数据
        scaler = MinMaxScaler()
        subset_scaled = scaler.fit_transform(subset[['Containerized Vessel SWT (Gen) (kg)']])

        # 创建序列
        try:
            X, y = create_sequences(subset_scaled, seq_length)
        except ValueError as ve:
            print(f"Error creating sequences for {port} - {commodity}: {ve}")
            continue

        # 将数据分为训练和测试集
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        print(f"Training model for {port} - {commodity}")

        try:
            with strategy.scope():
                # 进行超参数搜索
                tuner.search(X_train, y_train, epochs=100, validation_split=0.2, callbacks=callbacks,
                             batch_size=batch_size)

                # 获取最佳模型
                best_model = tuner.get_best_models(num_models=1)[0]

                # 保存模型和归一化参数
                best_model.save(f'models/{port}_{commodity}.keras')
                np.save(f'models/{port}_{commodity}_scaler.npy', scaler.scale_)
                print(f"Model for {port} - {commodity} saved successfully")

        except Exception as e:
            print(f"Error training model for {port} - {commodity}: {e}")