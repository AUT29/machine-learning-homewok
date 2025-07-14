import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random

# 加载训练和测试数据
train_df = pd.read_csv("../data/data/train.csv")
test_df = pd.read_csv("../data/data/test.csv")

# 数据预处理
train_df = train_df.drop(columns=["Date"])
test_df = test_df.drop(columns=["Date"])
data_scaler = MinMaxScaler()
scaled_train = data_scaler.fit_transform(train_df)
scaled_test = data_scaler.transform(test_df)


# 生成时间序列样本
def generate_samples(data, hist_len, pred_len):
    X, y = [], []
    total_points = len(data)
    for i in range(total_points - hist_len - pred_len + 1):
        X.append(data[i:i + hist_len, :-1])
        y.append(data[i + hist_len:i + hist_len + pred_len, -1])
    return np.array(X), np.array(y)


hist_window = 90  # 历史数据长度
pred_window = 90  # 预测长度
train_X, train_y = generate_samples(scaled_train, hist_window, pred_window)
test_X, test_y = generate_samples(scaled_test, hist_window, pred_window)

# 转换为PyTorch张量
train_X = torch.FloatTensor(train_X)
train_y = torch.FloatTensor(train_y)
test_X = torch.FloatTensor(test_X)
test_y = torch.FloatTensor(test_y)


# 自定义数据集
class SeqDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]


# 创建数据加载器
batch = 64
train_set = SeqDataset(train_X, train_y)
test_set = SeqDataset(test_X, test_y)
train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch, shuffle=False)


# 位置编码组件
class PosEncoding(nn.Module):
    def __init__(self, embed_size, max_seq=5000):
        super().__init__()
        pos = torch.arange(max_seq).unsqueeze(1)
        div = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(max_seq, 1, embed_size)
        pe[:, 0, 0::2] = torch.sin(pos * div)
        pe[:, 0, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


# Transformer编码器
class TransEncoder(nn.Module):
    def __init__(self, input_dim, embed_size=128, heads=8, layers=2):
        super().__init__()
        self.pos_enc = PosEncoding(embed_size)
        self.embed = nn.Linear(input_dim, embed_size)
        encoder = nn.TransformerEncoderLayer(embed_size, heads, dim_feedforward=512)
        self.encoder = nn.TransformerEncoder(encoder, layers)

    def forward(self, x):
        x = self.embed(x)
        x = x.permute(1, 0, 2)
        x = self.pos_enc(x)
        encoded = self.encoder(x)
        encoded = encoded.permute(1, 0, 2)
        return encoded[:, -1, :]


# LSTM编码器
class LSTMEnc(nn.Module):
    def __init__(self, input_dim, hidden_size=128, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, layers, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]


# 交叉注意力机制
class CrossAtt(nn.Module):
    def __init__(self, embed_size=128):
        super().__init__()
        self.q = nn.Linear(embed_size, embed_size)
        self.k = nn.Linear(embed_size, embed_size)
        self.v = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        q = self.q(x1).unsqueeze(1)
        k = self.k(x2).unsqueeze(1)
        v = self.v(x2).unsqueeze(1)
        scores = torch.bmm(q, k.transpose(1, 2)) / (128 ** 0.5)
        att = self.softmax(scores)
        out = torch.bmm(att, v)
        return out.squeeze(1)


# 融合模型
class HybridModel(nn.Module):
    def __init__(self, input_dim, output_dim=pred_window):
        super().__init__()
        self.trans = TransEncoder(input_dim)
        self.lstm = LSTMEnc(input_dim)
        self.att1 = CrossAtt()
        self.att2 = CrossAtt()
        self.fc = nn.Linear(256, output_dim)

    def forward(self, x):
        trans_out = self.trans(x)
        lstm_out = self.lstm(x)
        trans_att = self.att1(trans_out, lstm_out)
        lstm_att = self.att2(lstm_out, trans_out)
        combined = torch.cat([trans_att, lstm_att], dim=-1)
        return self.fc(combined)


# 训练准备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hybrid_model = HybridModel(train_X.shape[2]).to(device)
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(hybrid_model.parameters(), lr=0.0001)

# 模型训练
epochs = 500
loss_history = []

for epoch in range(epochs):
    hybrid_model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        opt.zero_grad()
        preds = hybrid_model(inputs)
        loss = loss_fn(preds, targets)
        loss.backward()
        opt.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    loss_history.append(avg_loss)
    if (epoch + 1) % 10 == 0:
        print(f'训练轮次 [{epoch + 1}/{epochs}], 损失值: {avg_loss:.4f}')

# 绘制训练曲线
plt.figure(figsize=(12, 6))
plt.plot(range(2, len(loss_history) + 1), loss_history[1:], 'b-', linewidth=2)
plt.xlabel('训练轮次', fontsize=14)
plt.ylabel('损失值', fontsize=14)
plt.title(f'混合模型训练损失 ({pred_window}天预测)', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(True)
plt.savefig(f'混合模型_{pred_window}天_训练损失.png', dpi=300)
plt.close()

# 模型评估
mse_list = []
mae_list = []
eval_runs = 10

for run in range(eval_runs):
    rand_idx = random.randint(0, len(test_X) - 1)
    sample_X = test_X[rand_idx:rand_idx + 1].to(device)
    sample_y = test_y[rand_idx:rand_idx + 1].to(device)

    with torch.no_grad():
        pred = hybrid_model(sample_X).cpu().numpy()[0]
    true = sample_y[0].cpu().numpy()

    mse = mean_squared_error(true, pred)
    mae = mean_absolute_error(true, pred)
    mse_list.append(mse)
    mae_list.append(mae)


    # 反归一化
    def denorm(pred, sample):
        temp = np.zeros((pred_window, sample.shape[2] + 1))
        temp[:, :-1] = np.tile(sample[0, -1, :].cpu().numpy(), (pred_window, 1))
        temp[:, -1] = pred
        return data_scaler.inverse_transform(temp)[:, -1]


    pred_denorm = denorm(pred, sample_X)
    true_denorm = denorm(true, sample_X)

    # 绘制预测图
    plt.figure(figsize=(15, 6))
    plt.plot(pred_denorm, 'b-', label='预测值')
    plt.plot(true_denorm, 'r--', label='真实值')
    plt.xlabel('时间步', fontsize=12)
    plt.ylabel('数值', fontsize=12)
    plt.title(f'混合模型预测结果 ({pred_window}天预测)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig(f'混合模型_{pred_window}天_预测_{run + 1}.png', dpi=300)
    plt.close()

# 输出评估结果
print(f'\n混合模型 {pred_window}天预测评估:')
for i in range(eval_runs):
    print(f'第{i + 1}次测试 - MSE误差: {mse_list[i]:.4f}, MAE误差: {mae_list[i]:.4f}')

avg_mse = np.mean(mse_list)
std_mse = np.std(mse_list)
avg_mae = np.mean(mae_list)
std_mae = np.std(mae_list)

print(f'\n综合评估指标:')
print(f'MSE均值: {avg_mse:.4f}, MSE标准差: {std_mse:.4f}')
print(f'MAE均值: {avg_mae:.4f}, MAE标准差: {std_mae:.4f}')