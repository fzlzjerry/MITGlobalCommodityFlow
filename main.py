import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LassoCV
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization


# EarlyStopping 类定义
class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


# 读取数据
data = pd.read_csv('/mnt/data/Port_level_Imports.csv')

# 数据预处理
data['Time'] = pd.to_datetime(data['Time'])
data.fillna(method='ffill', inplace=True)

# 数据归一化
scaler = MinMaxScaler()
data[['Containerized Vessel SWT (Gen) (kg)', 'Customs Containerized Vessel Value (Gen) ($US)']] = scaler.fit_transform(
    data[['Containerized Vessel SWT (Gen) (kg)', 'Customs Containerized Vessel Value (Gen) ($US)']]
)

# 特征工程
data['Month'] = data['Time'].dt.month
data['Year'] = data['Time'].dt.year
data['Quarter'] = data['Time'].dt.quarter
data['Day'] = data['Time'].dt.day
data['Week'] = data['Time'].dt.isocalendar().week
data['Weekday'] = data['Time'].dt.weekday
data['IsWeekend'] = data['Weekday'] >= 5

# 创建更多滞后特征
for lag in range(1, 25):
    data[f'Lag_{lag}'] = data['Containerized Vessel SWT (Gen) (kg)'].shift(lag)
data.dropna(inplace=True)

# 特征选择
features = ['Month', 'Year', 'Quarter', 'Day', 'Week', 'Weekday', 'IsWeekend',
            'Customs Containerized Vessel Value (Gen) ($US)'] + [f'Lag_{lag}' for lag in range(1, 25)]
target = 'Containerized Vessel SWT (Gen) (kg)'

X = data[features].values
y = data[target].values

lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected_features = np.where(lasso.coef_ != 0)[0]
X_selected = X[:, selected_features]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 创建数据集
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 构建 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, output_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dropout=dropout,
                                                        activation='gelu')
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, src):
        src = src.unsqueeze(1)  # (batch_size, seq_len, input_dim)
        transformer_out = self.transformer(src)  # (seq_len, batch_size, input_dim)
        transformer_out = transformer_out.squeeze(1)  # (batch_size, input_dim)
        output = self.fc(transformer_out)  # (batch_size, output_dim)
        output = self.dropout(output)
        output = self.fc2(output)
        return output


# 定义设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 使用贝叶斯优化来调参
def train_evaluate_model(nhead, num_layers, dropout, lr):
    nhead = int(nhead)
    num_layers = int(num_layers)
    dropout = max(min(dropout, 1), 0)
    lr = max(lr, 1e-6)

    model = TransformerModel(input_dim=len(selected_features), nhead=nhead, num_layers=num_layers, output_dim=1,
                             dropout=dropout)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10, delta=0)

    for epoch in range(50):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        scheduler.step(val_loss)
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            break

    return -val_loss


optimizer = BayesianOptimization(
    f=train_evaluate_model,
    pbounds={
        'nhead': (4, 16),
        'num_layers': (2, 8),
        'dropout': (0.1, 0.5),
        'lr': (1e-4, 1e-2)
    },
    random_state=42,
    verbose=2
)

optimizer.maximize(init_points=5, n_iter=10)

# 使用最佳参数重新训练模型
best_params = optimizer.max['params']
best_model = TransformerModel(
    input_dim=len(selected_features),
    nhead=int(best_params['nhead']),
    num_layers=int(best_params['num_layers']),
    output_dim=1,
    dropout=best_params['dropout']
)
best_model = nn.DataParallel(best_model, device_ids=[0, 1, 2, 3])
best_model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(best_model.parameters(), lr=best_params['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
early_stopping = EarlyStopping(patience=10, delta=0)

for epoch in range(100):
    best_model.train()
    running_loss = 0.0
    for batch in train_loader:
        optimizer.zero_grad()
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = best_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}')

    best_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = best_model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    print(f'Validation Loss: {val_loss / len(test_loader)}')

    scheduler.step(val_loss)
    early_stopping(val_loss, best_model)

    if early_stopping.early_stop:
        print("Early stopping")
        break

# 加载最佳模型
best_model.load_state_dict(torch.load('checkpoint.pt'))

# 模型评估
best_model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for batch in test_loader:
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = best_model(inputs)
        y_pred.append(outputs.cpu().numpy())
        y_true.append(targets.cpu().numpy())

y_pred = np.concatenate(y_pred, axis=0)
y_true = np.concatenate(y_true, axis=0)

# 计算评估指标
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
print(f'MSE: {mse}, MAE: {mae}, R^2: {r2}, MAPE: {mape}%')

# 可视化预测结果
plt.figure(figsize=(10, 6))
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
