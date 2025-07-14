import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

# 加载训练和测试数据
train_df = pd.read_csv("../data/data/train.csv")
test_df = pd.read_csv("../data/data/test.csv")

# 移除日期列并标准化数据
train_df = train_df.drop("Date", axis=1)
test_df = test_df.drop("Date", axis=1)
scaler = MinMaxScaler()
train_norm = scaler.fit_transform(train_df)
test_norm = scaler.transform(test_df)


# 生成序列数据
def make_sequences(data, in_len, out_len):
    features, targets = [], []
    total_len = len(data)
    for start in range(total_len - in_len - out_len + 1):
        end = start + in_len
        features.append(data[start:end, :-1])
        targets.append(data[end:end + out_len, -1])
    return np.array(features), np.array(targets)


seq_in = 90  # 输入序列长度
seq_out = 90  # 预测序列长度
train_X, train_y = make_sequences(train_norm, seq_in, seq_out)
test_X, test_y = make_sequences(test_norm, seq_in, seq_out)

# 转换为PyTorch张量
train_X = torch.FloatTensor(train_X)
train_y = torch.FloatTensor(train_y)
test_X = torch.FloatTensor(test_X)
test_y = torch.FloatTensor(test_y)


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# 创建数据加载器
batch = 32
train_set = CustomDataset(train_X, train_y)
test_set = CustomDataset(test_X, test_y)
train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)


# 定义lstm模型
class SeqPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, layers=2, output_dim=seq_out):
        super().__init__()
        self.lstm_layer = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_output, _ = self.lstm_layer(x)
        return self.output_layer(lstm_output[:, -1, :])


# 设置训练设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
predictor = SeqPredictor(train_X.shape[2]).to(device)
loss_func = nn.MSELoss()
adam_optim = torch.optim.Adam(predictor.parameters(), lr=0.001)

# 训练过程
epochs = 500
loss_history = []

for epoch in range(epochs):
    predictor.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        adam_optim.zero_grad()
        predictions = predictor(inputs)
        loss = loss_func(predictions, targets)
        loss.backward()
        adam_optim.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f'当前轮次 [{epoch + 1}/{epochs}], 损失值: {avg_loss:.4f}')

# 绘制训练损失曲线
plt.figure(figsize=(12, 6))
plt.plot(range(2, len(loss_history) + 1), loss_history[1:], 'b-', linewidth=2)
plt.xlabel('训练轮次', fontsize=14)
plt.ylabel('损失值', fontsize=14)
plt.title(f'LSTM训练损失曲线 ({seq_out}天预测)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig(f'./LSTM_{seq_out}_训练损失.png', dpi=300)
plt.close()

# 评估模型性能
mse_results = []
mae_results = []
test_runs = 10

for run in range(test_runs):
    # 随机选择测试样本
    sample_idx = random.randint(0, len(test_X) - 1)
    sample_X = test_X[sample_idx:sample_idx + 1].to(device)
    sample_y = test_y[sample_idx:sample_idx + 1].to(device)

    with torch.no_grad():
        pred = predictor(sample_X).cpu().numpy()[0]
    actual = sample_y[0].cpu().numpy()

    # 计算指标
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)
    mse_results.append(mse)
    mae_results.append(mae)


    # 反归一化数据
    def inverse_transform(pred, sample_X):
        temp = np.zeros((seq_out, sample_X.shape[2] + 1))
        temp[:, :-1] = np.tile(sample_X[0, -1, :].cpu().numpy(), (seq_out, 1))
        temp[:, -1] = pred
        return scaler.inverse_transform(temp)[:, -1]


    pred_actual = inverse_transform(pred, sample_X)
    true_actual = inverse_transform(actual, sample_X)

    # 绘制预测对比图
    plt.figure(figsize=(15, 6))
    plt.plot(pred_actual, 'b-', label='预测值')
    plt.plot(true_actual, 'r--', label='真实值')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.title(f'LSTM预测结果对比 ({seq_out}天预测)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f'LSTM_{seq_out}_预测结果_{run + 1}.png', dpi=300)
    plt.close()

# 输出评估结果
print(f'\nLSTM模型 {seq_out}天预测评估结果:')
for i in range(test_runs):
    print(f'第{i + 1}次测试 - MSE误差: {mse_results[i]:.4f}, MAE误差: {mae_results[i]:.4f}')

avg_mse = np.mean(mse_results)
std_mse = np.std(mse_results)
avg_mae = np.mean(mae_results)
std_mae = np.std(mae_results)

print(f'\n综合评估指标:')
print(f'MSE平均值: {avg_mse:.4f}, MSE标准差: {std_mse:.4f}')
print(f'MAE平均值: {avg_mae:.4f}, MAE标准差: {std_mae:.4f}')