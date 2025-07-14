import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

# 数据加载与预处理
df_train = pd.read_csv("../data/data/train.csv")
df_test = pd.read_csv("../data/data/test.csv")

# 移除日期列并进行归一化处理
df_train = df_train.drop(columns=["Date"])
df_test = df_test.drop(columns=["Date"])
data_normalizer = MinMaxScaler()
norm_train = data_normalizer.fit_transform(df_train)
norm_test = data_normalizer.transform(df_test)


# 生成时间序列数据
def generate_time_series(data, lookback, forecast):
    features, targets = [], []
    total_samples = len(data)
    for i in range(total_samples - lookback - forecast + 1):
        features.append(data[i:i + lookback, :-1])
        targets.append(data[i + lookback:i + lookback + forecast, -1])
    return np.array(features), np.array(targets)


lookback_window = 90  # 历史数据窗口大小
forecast_horizon = 90  # 预测时间跨度
train_features, train_targets = generate_time_series(norm_train, lookback_window, forecast_horizon)
test_features, test_targets = generate_time_series(norm_test, lookback_window, forecast_horizon)

# 转换为PyTorch张量
train_X = torch.FloatTensor(train_features)
train_y = torch.FloatTensor(train_targets)
test_X = torch.FloatTensor(test_features)
test_y = torch.FloatTensor(test_targets)


# 自定义数据集类
class TemporalDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


# 创建数据加载器
batch_size = 64
train_set = TemporalDataset(train_X, train_y)
test_set = TemporalDataset(test_X, test_y)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# 位置编码模块
class PositionEmbedding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=5000):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pos_embed = torch.zeros(max_seq_len, 1, embed_dim)
        pos_embed[:, 0, 0::2] = torch.sin(position * div_term)
        pos_embed[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embed', pos_embed)

    def forward(self, x):
        return x + self.pos_embed[:x.size(0)]


# Transformer模型定义
class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim=128, num_heads=8, num_layers=2, output_dim=forecast_horizon):
        super().__init__()
        self.position_embed = PositionEmbedding(embed_dim)
        self.feature_embed = nn.Linear(input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.predictor = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        x = self.feature_embed(x)
        x = x.permute(1, 0, 2)
        x = self.position_embed(x)
        encoded = self.encoder(x)
        return self.predictor(encoded[-1])


# 模型训练准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer_model = TemporalTransformer(train_X.shape[2]).to(device)
loss_function = nn.MSELoss()
model_optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)

# 训练过程
total_epochs = 500
training_loss = []

for epoch in range(total_epochs):
    transformer_model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        model_optimizer.zero_grad()
        predictions = transformer_model(batch_x)
        loss = loss_function(predictions, batch_y)
        loss.backward()
        model_optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    training_loss.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f'训练轮次 [{epoch + 1}/{total_epochs}], 损失值: {avg_loss:.4f}')

# 绘制训练损失曲线
plt.figure(figsize=(12, 6))
plt.plot(range(2, len(training_loss) + 1), training_loss[1:], 'b-', linewidth=2)
plt.xlabel('训练轮次', fontsize=14)
plt.ylabel('损失值', fontsize=14)
plt.title(f'Transformer训练损失曲线 ({forecast_horizon}天预测)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig(f'./Transformer_{forecast_horizon}天_训练损失.png', dpi=300)
plt.close()

# 模型评估
mse_values = []
mae_values = []
num_tests = 10

for test_num in range(num_tests):
    # 随机选择测试样本
    random_idx = random.randint(0, len(test_X) - 1)
    test_sample_X = test_X[random_idx:random_idx + 1].to(device)
    test_sample_y = test_y[random_idx:random_idx + 1].to(device)

    with torch.no_grad():
        pred = transformer_model(test_sample_X).cpu().numpy()[0]
    actual = test_sample_y[0].cpu().numpy()

    # 计算评估指标
    current_mse = mean_squared_error(actual, pred)
    current_mae = mean_absolute_error(actual, pred)
    mse_values.append(current_mse)
    mae_values.append(current_mae)


    # 反归一化处理
    def denormalize(prediction, sample):
        temp_array = np.zeros((forecast_horizon, sample.shape[2] + 1))
        temp_array[:, :-1] = np.tile(sample[0, -1, :].cpu().numpy(), (forecast_horizon, 1))
        temp_array[:, -1] = prediction
        return data_normalizer.inverse_transform(temp_array)[:, -1]


    denorm_pred = denormalize(pred, test_sample_X)
    denorm_actual = denormalize(actual, test_sample_X)

    # 绘制预测对比图
    plt.figure(figsize=(15, 6))
    plt.plot(denorm_pred, 'b-', label='预测值')
    plt.plot(denorm_actual, 'r--', label='真实值')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.title(f'Transformer预测结果对比 ({forecast_horizon}天预测)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f'./Transformer_{forecast_horizon}天_预测结果_{test_num + 1}.png', dpi=300)
    plt.close()

# 输出评估结果
print(f'\nTransformer模型 {forecast_horizon}天预测评估:')
for i in range(num_tests):
    print(f'第{i+1}测试 - MSE误差: {mse_values[i]:.4f}, MAE误差: {mae_values[i]:.4f}')

avg_mse = np.mean(mse_values)
mse_std = np.std(mse_values)
avg_mae = np.mean(mae_values)
mae_std = np.std(mae_values)

print(f'\n综合评估结果:')
print(f'MSE平均值: {avg_mse:.4f}, MSE标准差: {mse_std:.4f}')
print(f'MAE平均值: {avg_mae:.4f}, MAE标准差: {mae_std:.4f}')