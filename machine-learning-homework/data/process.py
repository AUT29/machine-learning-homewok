# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 读取数据
# df = pd.read_csv("train.csv", parse_dates=['DateTime'])  # 请替换为你实际的文件名和时间列名

# # 确保时间戳列是 datetime 类型
# df['DateTime'] = pd.to_datetime(df['DateTime'])

# # 筛选最近一个月的数据（假设数据从现在开始往前一个月）
# latest_month = df[df['DateTime'] >= df['DateTime'].max() - pd.Timedelta(days=30)]

# # 提取小时字段
# latest_month['hour'] = latest_month['DateTime'].dt.hour
# plt.figure(figsize=(12, 6))
# sns.boxplot(x='hour', y='Global_active_power', data=latest_month, palette='Set2')
# plt.title('Hourly Distribution of Total Active Power (Past 30 Days)')
# plt.xlabel('Hour of Day')
# plt.ylabel('Total Active Power (kW)')
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('xiaoshi.png')
import pandas as pd
import matplotlib.pyplot as plt

# # 读取数据，并将 "?" 等非法值处理为 NaN
# df = pd.read_csv("train.csv", parse_dates=['DateTime'], na_values=["?"])
# df['DateTime'] = pd.to_datetime(df['DateTime'])

# # 将 Global_active_power 转为 float 类型（如果还不是）
# df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# # 筛选 2007 年 1 月的数据
# df_jan2007 = df[(df['DateTime'] >= '2007-01-01') & (df['DateTime'] < '2007-02-01')]

# # 按小时聚合（取平均），自动跳过 NaN
# df_hourly = df_jan2007.set_index('DateTime').resample('H')['Global_active_power'].mean().reset_index()

# # 画图
# plt.figure(figsize=(18, 6))
# plt.plot(df_hourly['DateTime'], df_hourly['Global_active_power'], color='blue', linewidth=1)
# plt.title('Hourly Global Active Power - January 2007')
# plt.xlabel('Time')
# plt.ylabel('Global Active Power (kW)')
# plt.xticks(rotation=45)
# plt.grid(True)
# plt.tight_layout()
# plt.savefig('xiaoshi.png')

# 读取数据，并将 "?" 等非法值处理为 NaN
df = pd.read_csv("data/test.csv", parse_dates=['Date'], na_values=["?"])
df['Date'] = pd.to_datetime(df['Date'])

# 将 Global_active_power 转为 float 类型（如果还不是）
df['RR'] = pd.to_numeric(df['RR'], errors='coerce')

# 筛选 2007 年 1 月的数据
# df_jan2007 = df[(df['Date'] >= '2007-01-01') & (df['Date'] < '2008-01-01')]

# 按小时聚合（取平均），自动跳过 NaN
# df_hourly = df_jan2007.set_index('Date')['Global_active_power'].mean()

# 画图
plt.figure(figsize=(18, 6))
plt.plot(df['Date'], df['RR'], color='blue', linewidth=1)
plt.title('Daily  Global_intensity - test')
plt.xlabel('Time')
plt.ylabel('Global Active Power (kW)')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('test_tian_Global_intensity.png')