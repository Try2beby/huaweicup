import ast
import re

import numpy as np
import pandas as pd
import pywt
import scipy.stats as stats
from scipy.fft import fft

df_2ap = pd.read_csv("../data/test_set_2_2ap.csv")


# 合并 ap 间数据
def merge_rssi_columns(row, prefix):
    """
    合并 RSSI 列，根据 ap_id 或 sta_id 选择合适的列进行合并。
    prefix 是 'ap_from_ap' 或 'sta_to_ap' 等
    """
    if row["ap_id"] == 0:
        sum_rssi = row[f"{prefix}_0_sum_ant_rssi"]
        max_rssi = row[f"{prefix}_0_max_ant_rssi"]
        mean_rssi = row[f"{prefix}_0_mean_ant_rssi"]
    else:
        sum_rssi = row[f"{prefix}_1_sum_ant_rssi"]
        max_rssi = row[f"{prefix}_1_max_ant_rssi"]
        mean_rssi = row[f"{prefix}_1_mean_ant_rssi"]

    return pd.Series(
        [sum_rssi, max_rssi, mean_rssi],
        index=[f"{prefix}_sum_rssi", f"{prefix}_max_rssi", f"{prefix}_mean_rssi"],
    )


mask = np.array(df_2ap["ap_id"] == "ap_1")

# 应用到df_2ap
df_2ap[["ap_from_ap_sum_rssi", "ap_from_ap_max_rssi", "ap_from_ap_mean_rssi"]] = (
    np.where(
        np.repeat(mask[:, np.newaxis], 3, axis=1),
        df_2ap[
            [
                "ap_from_ap_0_sum_ant_rssi",
                "ap_from_ap_0_max_ant_rssi",
                "ap_from_ap_0_mean_ant_rssi",
            ]
        ],
        df_2ap[
            [
                "ap_from_ap_1_sum_ant_rssi",
                "ap_from_ap_1_max_ant_rssi",
                "ap_from_ap_1_mean_ant_rssi",
            ]
        ],
    )
)
# 合并sta间数据
df_2ap["sta_from_sta_rssi"] = np.where(
    mask, df_2ap["sta_from_sta_0_rssi"], df_2ap["sta_from_sta_1_rssi"]
)


def transform_rssi(row):
    bss_id = row["bss_id"]  # Assuming 'bss_id' is a column in your DataFrame
    if bss_id != 0:
        return row

    _sum_rssi = ast.literal_eval(row["sta_from_ap_0_sum_ant_rssi"])
    row["sta_from_ap_0_mean_ant_rssi"] = str([num - 9 for num in _sum_rssi])
    row["sta_from_ap_0_max_ant_rssi"] = str([num - 9 + 3 for num in _sum_rssi])

    return row


df_2ap = df_2ap.apply(transform_rssi, axis=1)

df_2ap["mcs_nss"] = df_2ap.apply(lambda row: None, axis=1)
df_2ap.drop(columns=["mcs", "nss"], inplace=True)


def approximate_entropy(U, m, r):
    """计算近似熵 (Approximate Entropy)"""
    if len(U) < m + 1:
        return np.nan  # 数据不足，无法计算近似熵

    def _phi(m):
        x = np.array([U[i : i + m] for i in range(len(U) - m + 1)])
        C = np.sum(np.abs(x[:, None] - x[None, :]).max(axis=2) <= r, axis=0) / (
            len(U) - m + 1
        )
        return np.log(C).sum() / (len(U) - m + 1)

    return _phi(m) - _phi(m + 1)


def grouping_entropy(column_data, num_bins):
    """分组熵"""
    if len(column_data) == 0:
        return np.nan
    hist, _ = np.histogram(column_data, bins=num_bins)
    probabilities = hist / len(column_data)
    probabilities = probabilities[probabilities > 0]  # 过滤掉零概率
    if len(probabilities) == 0:
        return np.nan  # 避免 log2(0) 的情况
    return -np.sum(probabilities * np.log2(probabilities))


def extract_statistics_for_column(row, column_name):
    """
    针对单个RSSI列的数据提取统计特征
    :param column_data: 某一列的数据，元素是列表
    :return: 统计特征的字典
    """
    column_data = row[column_name]

    column_data = ast.literal_eval(column_data)
    column_data_np = np.array(column_data)

    if len(column_data) == 0:
        return {"error": "empty data"}

    # pd_ = row["pd"]
    ed = row["ed"]
    nav = row["nav"]
    bss_id = row["bss_id"]
    statistics = {}

    if "mean" in column_name:
        statistics["le_nav_percent"] = np.mean(column_data_np <= nav)

        # if column_name starts with "sta_from_ap_"
        if column_name.startswith("sta_from_ap_"):
            _mean = np.mean(column_data_np)
            lis1 = ast.literal_eval(row[f"sta_from_ap_{bss_id}_mean_ant_rssi"])
            lis2 = ast.literal_eval(row[f"sta_from_ap_{1-bss_id}_mean_ant_rssi"])
            if _mean >= nav and _mean <= ed:
                sinr = np.mean(lis1) - np.mean(lis2)
            else:
                sinr = np.mean(lis1) - (-99)
            statistics["sinr"] = sinr

    elif "max" in column_name:
        # print(column_data_np >= nav)
        # print(column_data_np <= ed)
        statistics["in_nav_ed_percent"] = np.sum(
            np.logical_and(column_data_np >= nav, column_data_np <= ed)
        ) / len(column_data_np)
        statistics["la_ed_percent"] = np.mean(column_data_np >= ed)

    ### 基础统计量 ###
    statistics["org"] = column_data
    statistics["length"] = len(column_data)  # 数据长度
    statistics["max"] = np.max(column_data)  # 最大值
    statistics["min"] = np.min(column_data)  # 最小值
    statistics["median"] = np.median(column_data)  # 中位数
    statistics["range"] = statistics["max"] - statistics["min"]  # 范围
    statistics["iqr"] = np.percentile(column_data, 75) - np.percentile(
        column_data, 25
    )  # 四分位距
    statistics["mean"] = np.mean(column_data)  # 平均值
    statistics["var"] = np.var(column_data)  # 方差

    # 判断数据是否几乎相同，避免计算偏度和峰度时的精度丢失
    if np.var(column_data) < 1e-8:  # 设置一个非常小的阈值
        statistics["kurtosis"] = np.nan  # 跳过峰度计算
        statistics["skewness"] = np.nan  # 跳过偏度计算
    else:
        try:
            statistics["kurtosis"] = stats.kurtosis(column_data)  # 峰度
            statistics["skewness"] = stats.skew(column_data)  # 偏度
        except RuntimeWarning:
            statistics["kurtosis"] = np.nan
            statistics["skewness"] = np.nan

    if len(column_data) > 1:
        statistics["rate_of_change"] = np.diff(column_data).mean()  # 变化率
        statistics["sum_absolute_diff"] = np.sum(
            np.abs(np.diff(column_data))
        )  # 差分绝对和
    else:
        statistics["rate_of_change"] = np.nan
        statistics["sum_absolute_diff"] = np.nan

    ### 复杂统计量 ###
    # 检查数据点数是否足够计算回归
    if len(column_data) > 1:
        time = np.arange(len(column_data))
        try:
            slope, intercept, _, _, _ = stats.linregress(time, column_data)
            statistics["trend"] = slope  # 信号的趋势
        except RuntimeWarning:
            statistics["trend"] = np.nan
    else:
        statistics["trend"] = np.nan

    # Entropy
    value_counts = np.unique(column_data, return_counts=True)[1]
    probabilities = value_counts / len(column_data)
    if len(probabilities) > 0:
        statistics["entropy"] = -np.sum(probabilities * np.log2(probabilities))  # 熵
    else:
        statistics["entropy"] = np.nan

    # # SNR
    # signal_power = np.mean(np.square(column_data))
    # noise_power = np.var(column_data)
    # statistics["snr_"] = (
    #     signal_power / noise_power if noise_power != 0 else np.nan
    # )  # 信噪比

    # 自相关系数
    if len(column_data) > 1:
        statistics["autocorrelation"] = np.corrcoef(column_data[:-1], column_data[1:])[
            0, 1
        ]  # 自相关系数
    else:
        statistics["autocorrelation"] = np.nan

    # Approximate Entropy
    statistics["approximate_entropy"] = approximate_entropy(
        column_data, 2, 0.2 * np.std(column_data)
    )  # 近似熵

    # Grouping Entropy
    statistics["grouping_entropy"] = grouping_entropy(column_data, 10)  # 分组熵

    ### 频域统计量 ###
    # Fourier Coefficients
    if len(column_data) > 1:
        fft_coefficients = np.abs(fft(column_data))
        statistics["fourier_coefficients"] = np.mean(fft_coefficients)  # 傅里叶系数均值
    else:
        statistics["fourier_coefficients"] = np.nan

    # Wavelet Transform
    def deal_coeff(coeffs):
        # compute l2 norm for each list in coeffs
        l2_norm = [np.linalg.norm(c) for c in coeffs]
        # return mean of l2 norm
        return np.mean(l2_norm)

    if len(column_data) > 1:
        coeffs = pywt.wavedec(column_data, "db1")
        statistics["wavelet_coefficients"] = deal_coeff(coeffs)
    else:
        statistics["wavelet_coefficients"] = np.nan

    # fill nan with 0
    for key in statistics:
        if key != "org" and pd.isna(statistics[key]):
            statistics[key] = 0
    return statistics


# 找出所有需要提取统计特征的RSSI列
rssi_columns = [
    col
    for col in df_2ap.columns
    if col.endswith("rssi") and not re.match(r"sta_from_sta|ap_from_ap_[01]", col)
]

print(f"rssi_columns: {rssi_columns}")

# 处理每个 RSSI 列
all_statistics = {}
new_columns_tupled = []
for col in rssi_columns:
    # 对每个 RSSI 列进行统计特征提取
    stats_d = df_2ap.apply(
        lambda row: extract_statistics_for_column(row, col), axis=1
    )  # 逐行提取统计特征

    # 将提取的统计特征展开并作为子列添加
    all_statistics[col] = pd.DataFrame(stats_d.tolist(), index=df_2ap.index)

    new_columns_tupled.extend([(col, stat) for stat in all_statistics[col].columns])

df_2ap.drop(columns=rssi_columns, inplace=True)

raw_columns_tupled = [(col, "_") for col in df_2ap.columns]
# 合并统计特征
for col, stats_df in all_statistics.items():
    df_2ap = pd.concat([df_2ap, stats_df.add_prefix(f"{col}_")], axis=1)

df_2ap.columns = pd.MultiIndex.from_tuples(raw_columns_tupled + new_columns_tupled)

# assert len(new_columns_tupled) == len(rssi_columns) * len(
#     all_statistics[rssi_columns[0]].columns
# )

# ('sta_to_ap_0_sum_ant_rssi', 'org')
# drop columns with "sum" in name[0]
df_2ap.drop(columns=[col for col in df_2ap.columns if "sum" in col[0]], inplace=True)

# print(len(df_2ap.columns))
# print(f"# of columns: {len(df_2ap.columns)}")


# (*rssi, !org)
# (!*rssi, _)

# 提取与 "*rssi, !org" 相关的列 (一级索引包含 'rssi' 且二级索引不等于 'org')
rssi_not_org_cols = [
    col
    for col in df_2ap.columns
    if "rssi" in col[0] and col[1] != "org" and col[1] != "_"
]

# 提取与 "!*rssi, _" 相关的列 (一级索引不包含 'rssi' 且二级索引等于 '_')
not_rssi_underscore_cols = [
    col for col in df_2ap.columns if "rssi" not in col[0] and col[1] == "_"
]

# 创建新的 DataFrame 包含提取出的列
df_rssi_not_org = df_2ap[rssi_not_org_cols]
df_not_rssi_underscore = df_2ap[not_rssi_underscore_cols]

cols_to_use = [
    # ("test_id", "_"),
    # ("test_dur", "_"),
    # ("loc_id", "_"),
    ("protocol", "_"),
    # ("pkt_len", "_"),
    ("bss_id", "_"),
    # ("ap_name", "_"),
    # ("ap_mac", "_"),
    # ("ap_id", "_"),
    # ("pd", "_"),
    # ("ed", "_"),
    ("nav", "_"),
    ("eirp", "_"),
    # ("sta_mac", "_"),
    # ("sta_id", "_"),
    # ("seq_time", "_"),
    ("mcs_nss", "_"),
]

catagory_cols = [
    # ("loc_id", "_"),
    ("protocol", "_"),
    ("bss_id", "_"),
    # ("ap_mac", "_"),
    # ("sta_mac", "_"),
]


df_2ap = pd.concat([df_not_rssi_underscore[cols_to_use], df_rssi_not_org], axis=1)

# # deal with catagory columns
# for col in catagory_cols:
#     df_2ap[col] = df_2ap[col].astype("category")


df_dummies = pd.get_dummies(df_2ap[catagory_cols])
new_cols = [(col, "_") for col in df_dummies.columns]
# new col names (org_name, _)
df_dummies.columns = pd.MultiIndex.from_tuples(new_cols)


df_2ap = pd.concat([df_2ap, df_dummies], axis=1)
df_2ap.drop(columns=catagory_cols, inplace=True)

print("# of columns: ", len(df_2ap.columns))

# save
df_2ap.to_csv("./df_2_2ap_test_final.csv", index=False)
