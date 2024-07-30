import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, BatchNormalization, Conv1D, MultiHeadAttention, \
    LayerNormalization, Add
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.regularizers import l1_l2
import keras_tuner as kt
import os
import joblib

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

# 添加时间特征
data['Month'] = data['Time'].dt.month
data['Quarter'] = data['Time'].dt.quarter
data['Year'] = data['Time'].dt.year
data['DayOfWeek'] = data['Time'].dt.dayofweek
data['IsWeekend'] = data['DayOfWeek'] >= 5


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


class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        input_layer = Input(shape=(seq_length, data.shape[1] - 1))

        x = input_layer
        for i in range(10):
            x = Conv1D(filters=hp.Int(f'filters_conv_{i}', min_value=32, max_value=256, step=32),
                       kernel_size=hp.Int(f'kernel_size_conv_{i}', min_value=2, max_value=5, step=1),
                       activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(rate=hp.Float(f'dropout_conv_{i}', min_value=0.2, max_value=0.5, step=0.1))(x)

        for i in range(10):
            lstm_layer = LSTM(units=hp.Int(f'units_lstm_{i}', min_value=128, max_value=1024, step=64),
                              return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
            x = Add()([x, lstm_layer])  # 残差连接
            x = BatchNormalization()(x)
            x = Dropout(rate=hp.Float(f'dropout_lstm_{i}', min_value=0.2, max_value=0.5, step=0.1))(x)

        for i in range(10):
            gru_layer = GRU(units=hp.Int(f'units_gru_{i}', min_value=128, max_value=1024, step=64),
                            return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
            x = Add()([x, gru_layer])  # 残差连接
            x = BatchNormalization()(x)
            x = Dropout(rate=hp.Float(f'dropout_gru_{i}', min_value=0.2, max_value=0.5, step=0.1))(x)

        multi_head_attention_layer = MultiHeadAttention(num_heads=8, key_dim=hp.Int('key_dim_attention', min_value=128,
                                                                                    max_value=512, step=64))(x, x)
        x = Add()([x, multi_head_attention_layer])  # 残差连接
        x = LayerNormalization()(x)
        x = Dropout(rate=hp.Float('dropout_attention', min_value=0.2, max_value=0.5, step=0.1))(x)

        for i in range(10):
            dense_layer = Dense(units=hp.Int(f'units_dense_{i}', min_value=128, max_value=512, step=64),
                                activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
            x = Dropout(rate=hp.Float(f'dropout_dense_{i}', min_value=0.2, max_value=0.5, step=0.1))(dense_layer)

        output_layer = Dense(1)(x)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'adamax', 'nadam']), loss='mse')
        return model

    def get_config(self):
        return {}

    def build_from_config(self, config):
        return self.build(config)


# 启用混合精度训练
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 设置超参数调整器
tuner = kt.Hyperband(
    MyHyperModel(),
    objective='val_loss',
    max_epochs=200,
    factor=3,
    directory='my_dir',
    project_name='commodity_forecasting'
)

# 参数
batch_size = 512

# 获取唯一的港口和商品
ports = data['Port'].unique()
commodities = data['Commodity'].unique()

# 早停、学习率减少和模型检查点的回调函数
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
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
            continue

        # 规范化数据
        scaler = MinMaxScaler()
        subset_scaled = scaler.fit_transform(
            subset[['Containerized Vessel SWT (Gen) (kg)', 'Month', 'Quarter', 'Year', 'DayOfWeek', 'IsWeekend']])

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
            # 进行超参数搜索
            tuner.search(X_train, y_train, epochs=200, validation_split=0.2, callbacks=callbacks, batch_size=batch_size)

            # 获取最佳模型
            best_model = tuner.get_best_models(num_models=1)[0]

            # 保存模型和归一化参数
            best_model.save(f'models/{port}_{commodity}.keras')
            joblib.dump(scaler, f'models/{port}_{commodity}_scaler.pkl')
            print(f"Model for {port} - {commodity} saved successfully")

        except Exception as e:
            print(f"Error training model for {port} - {commodity}: {e}")
