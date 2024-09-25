import pandas as pd
from modeling_old import clear_column_names, random_search
from sklearn.preprocessing import LabelEncoder


def load_data(path="./df_2ap_final.csv", threshold=1):
    df = pd.read_csv(path, header=[0, 1], index_col=None)

    y = df[("mcs_nss", "_")]
    X = df.drop(columns=[("mcs_nss", "_")])

    X.columns = clear_column_names(X)

    # 计算每个类别的频次
    value_counts = y.value_counts()
    # 找到频次大于阈值的类别
    valid_categories = value_counts[value_counts >= threshold].index

    # 过滤样本
    mask = y.isin(valid_categories)
    X = X[mask]
    y = y[mask]

    return X, y


X, y = load_data(threshold=5)  # 设置阈值为 5


# 创建 LabelEncoder 实例
le = LabelEncoder()

# 转换目标变量
y_encoded = le.fit_transform(y)

# to pd.Series
y_encoded = pd.Series(y_encoded)

model_types = dict(
    # random_forest=dict(
    #     n_estimators=[50, 100, 200],
    #     max_features=[None, "sqrt", "log2"],
    #     max_depth=[None, 10, 20],
    # ),
    # extra_trees=dict(
    #     n_estimators=[50, 100, 200],
    #     max_features=[None, "sqrt", "log2"],
    #     max_depth=[None, 10, 20],
    # ),
    xgboost=dict(
        objective="multi:softmax",
        n_estimators=[50, 100, 200],
        max_depth=[None, 10, 20],
        learning_rate=[0.01, 0.1, 0.2],
    ),
    lightgbm=dict(
        objective="multiclass",  # 多分类
        metric="multi_logloss",  # 多分类的损失函数
        n_estimators=[50, 100, 200],
        learning_rate=[0.01, 0.1, 0.2],
        max_depth=[None, 10, 20],
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        early_stopping_round=10,
    ),
    catboost=dict(
        iterations=[50, 100, 200],
        depth=[4, 6, 8],
        learning_rate=[0.01, 0.1, 0.2],
        l2_leaf_reg=[1, 3, 5],
        loss_function="MultiClass",  # 多分类
        bootstrap_type=["Bernoulli", "MVS"],
    ),
    mlp=dict(
        hidden_size1=[64, 128, 256],
        hidden_size2=[32, 64, 128],
        weight_decay=[0.001, 0.0001, 1e-5],
        output_size=14,  # 类别数量
    ),
)

results = {}
for k, v in model_types.items():
    print(f"Running {k}")
    res, _ = random_search(
        param_dict=v, X=X, y=y_encoded, model_type=k, suffix="3ap", n_trials=10
    )
    results[k] = res
