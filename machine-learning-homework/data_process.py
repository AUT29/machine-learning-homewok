import pandas as pd
import numpy as np

def preprocess_minute_to_daily(csv_path, save_path=None, dropna=True, min_minutes_per_day=1440):
    df = pd.read_csv(csv_path, parse_dates=['DateTime'])
    df['Date'] = df['DateTime'].dt.date

    # 转换为float
    cols_to_convert = [
        'Global_active_power', 'Global_reactive_power', 'Voltage',
        'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
        'RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU'
    ]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 可选丢弃含NaN行
    if dropna:
        df = df.dropna()

    # 仅保留分钟数够的完整天
    counts_per_day = df.groupby('Date').size()
    complete_days = counts_per_day[counts_per_day >= min_minutes_per_day].index
    df = df[df['Date'].isin(complete_days)]

    # 日聚合
    daily_df = df.groupby('Date').agg({
        'Global_active_power': 'sum',
        'Global_reactive_power': 'sum',
        'Sub_metering_1': 'sum',
        'Sub_metering_2': 'sum',
        'Sub_metering_3': 'sum',
        'Voltage': 'mean',
        'Global_intensity': 'mean',
        'RR': 'first',
        'NBJRR1': 'first',
        'NBJRR5': 'first',
        'NBJRR10': 'first',
        'NBJBROU': 'first'
    }).reset_index()

    # 填补缺失日期
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    daily_df = daily_df.set_index('Date').asfreq('D')  # 补全日期（NaN 的天）

    # ========== 缺失天填充逻辑 ==========
    sum_cols = ['Global_active_power', 'Global_reactive_power',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    mean_cols = ['Voltage', 'Global_intensity']
    random_cols = ['RR', 'NBJRR1', 'NBJRR5', 'NBJRR10', 'NBJBROU']

    for col in sum_cols + mean_cols:
        daily_df[col] = daily_df[col].interpolate(method='linear', limit_direction='both')

    # for col in random_cols:
    #     # 用前一天或后一天的值（或中值）来填充
    #     daily_df[col] = daily_df[col].fillna(method='ffill').fillna(method='bfill')
    # 遍历每列
    for col in random_cols:
        missing_dates = daily_df[daily_df[col].isna()].index
        for date in missing_dates:
            # 查找原始 df 中对应这一天的分钟数据
            values = df[df['Date'] == date.date()][col].dropna()
            if not values.empty:
                # 用当天的中位数 / 均值 / 首值 等你选择
                daily_df.loc[date, col] = values.median()  # 你也可以用 .mean() 或 .iloc[0]
            else:
                # 若原始数据该天也没有，则 fallback
                daily_df.loc[date, col] = daily_df[col].ffill().bfill().loc[date]
    # 补充 Sub_metering_remainder
    daily_df['Sub_metering_remainder'] = (
        daily_df['Global_active_power'] * 1000 / 60
        - daily_df['Sub_metering_1']
        - daily_df['Sub_metering_2']
        - daily_df['Sub_metering_3']
    )

    daily_df = daily_df.reset_index()  # 恢复日期为列

    if save_path:
        daily_df.to_csv(save_path, index=False)

    return daily_df



datapath = "data/train.csv"
save_path = "data/data/train.csv"
preprocess_minute_to_daily(datapath, save_path)
datapath = "data/test.csv"
save_path = "data/data/test.csv"
preprocess_minute_to_daily(datapath, save_path)