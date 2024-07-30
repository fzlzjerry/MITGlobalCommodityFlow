import pandas as pd
import numpy as np

# 读取数据
file_path = 'Port level Imports.csv'  # 修改为你的文件路径
data = pd.read_csv(file_path)

# 将时间列转换为日期时间格式
data['Time'] = pd.to_datetime(data['Time'])

# 将适当的列转换为数值类型
data['Containerized Vessel SWT (Gen) (kg)'] = data['Containerized Vessel SWT (Gen) (kg)'].str.replace(',', '').astype(
    float)
data['Customs Containerized Vessel Value (Gen) ($US)'] = data[
    'Customs Containerized Vessel Value (Gen) ($US)'].str.replace(',', '').astype(float)

# 重命名列
data = data.rename(columns={
    'Containerized Vessel SWT (Gen) (kg)': 'SWT',
    'Customs Containerized Vessel Value (Gen) ($US)': 'Value'
})

# 获取唯一的港口和商品组合
port_commodity_combinations = data[['Port', 'Commodity']].drop_duplicates()

# 创建存储补全数据的列表
filled_data_list = []

# 遍历每个港口和商品组合
for index, row in port_commodity_combinations.iterrows():
    port = row['Port']
    commodity = row['Commodity']

    # 筛选数据
    df_pc = data[(data['Port'] == port) & (data['Commodity'] == commodity)]

    # 删除无效数据 (0)
    df_pc = df_pc.query('SWT != 0 and Value != 0')

    # 设置时间索引并生成完整的月份范围
    df_pc = df_pc.set_index('Time')
    df_pc = df_pc[~df_pc.index.duplicated(keep='first')]  # 保留重复数据的第一次出现
    full_date_range = pd.date_range(start=df_pc.index.min(), end=df_pc.index.max(), freq='MS')  # 生成每月的开始日期
    df_pc = df_pc.reindex(full_date_range)

    # 确保对象类型列转换为适当类型
    df_pc = df_pc.infer_objects()

    # 补全非数值列（如有必要）
    # 检查是否存在需要插值的其他列
    non_numeric_cols = df_pc.select_dtypes(exclude=[np.number]).columns
    for col in non_numeric_cols:
        df_pc[col] = df_pc[col].ffill().bfill()

    # 仅对数值列进行插值
    numeric_cols = df_pc.select_dtypes(include=[np.number]).columns
    df_pc[numeric_cols] = df_pc[numeric_cols].interpolate(method='linear')
    df_pc = df_pc.reset_index().rename(columns={'index': 'Time'})

    # 添加到补全数据列表
    filled_data_list.append(df_pc)

# 合并所有补全数据
filled_data = pd.concat(filled_data_list, ignore_index=True)

# 保存补全后的数据
output_file_path = 'Port_level_Imports_Filled_All.csv'  # 修改为你的输出路径
filled_data.to_csv(output_file_path, index=False)
print(f"补全后的数据已保存到：{output_file_path}")
