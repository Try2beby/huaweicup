import pandas as pd
from modeling_old import clear_column_names, random_search


def load_data(path="./df_2ap_final.csv"):
    df = pd.read_csv(path, header=[0, 1], index_col=None)

    y = df[("throughput", "_")]
    X = df.drop(columns=[("throughput", "_")])

    X.columns = clear_column_names(X)

    return X, y


X, y = load_data("./df_3ap_final.csv")

model_types = dict(
    random_forest=dict(
        n_estimators=[50, 100, 200],
        max_features=[None, "sqrt", "log2"],
        max_depth=[None, 10, 20],
    ),
    extra_trees=dict(
        n_estimators=[50, 100, 200],
        max_features=[None, "sqrt", "log2"],
        max_depth=[None, 10, 20],
    ),
    xgboost=dict(
        objective="multi:softmax",  # 多分类
        num_class=15,  # 类别数量
        n_estimators=[50, 100, 200],
        max_depth=[3, 5, 7],
        learning_rate=[0.01, 0.1, 0.2],
    ),
    lightgbm=dict(
        objective="multiclass",  # 多分类
        metric="multi_logloss",  # 多分类的损失函数
        n_estimators=[50, 100, 200],
        learning_rate=[0.01, 0.1, 0.2],
        num_leaves=[31, 63, 127],
        min_data_in_leaf=[20, 50, 100],
        feature_fraction=[0.8, 0.9, 1.0],
        early_stopping_round=[10],
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
        param_dict=v, X=X, y=y, model_type=k, suffix="3ap", n_trials=10
    )
    results[k] = res
