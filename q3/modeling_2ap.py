import pandas as pd
from modeling import clear_column_names, regressor

df = pd.read_csv("./df_2ap_final.csv", header=[0, 1], index_col=None)

y = df[("seq_time", "_")]
X = df.drop(columns=[("seq_time", "_")])

X.columns = clear_column_names(X)


model_types = dict(
    random_forest=dict(
        n_estimators=[50, 100, 200],
        max_features=[None, "sqrt", "log2"],
        max_depth=[None, 10, 20],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        bootstrap=[True, False],
        # criterion=["gini", "entropy"],
    ),
    lightgbm=dict(
        n_estimators=[50, 100, 200],
        learning_rate=[0.01, 0.1, 0.2],
        max_depth=[-1, 10, 20],
        num_leaves=[31, 63, 127],
        min_data_in_leaf=[20, 50, 100],
        feature_fraction=[0.6, 0.8, 1.0],
        bagging_fraction=[0.6, 0.8, 1.0],
        bagging_freq=[1, 5, 10],
        lambda_l1=[0, 0.1, 1],
        lambda_l2=[0, 0.1, 1],
    ),
    extra_trees=dict(
        n_estimators=[50, 100, 200],
        max_features=[None, "sqrt", "log2"],
        max_depth=[None, 10, 20],
        min_samples_split=[2, 5, 10],
        min_samples_leaf=[1, 2, 4],
        bootstrap=[True, False],
    ),
    catboost=dict(
        dict(
            iterations=[50, 100, 200],
            depth=[4, 6, 8],
            learning_rate=[0.01, 0.1, 0.2],
            l2_leaf_reg=[1, 3, 5],
            loss_function=["RMSE", "MAE"],
            bootstrap_type=["Bayesian", "Bernoulli", "MVS"],
        )
    ),
    mlp=dict(
        hidden_size1=[64, 128, 256],
        hidden_size2=[32, 64, 128],
        lr=[0.001, 0.01, 0.1],
        batch_size=[32, 64, 128],
    ),
)
results = {}


for k, v in model_types.items():
    print(f"Running {k}")
    res = regressor(X, y, params=v, model_type=k)
    results[k] = res
