import random
import re

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from catboost import CatBoostRegressor
from matplotlib_inline import backend_inline
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

backend_inline.set_matplotlib_formats("svg")


def clear_column_names(df):
    columns = [re.sub(r"[\'\"(\s]", "", str(col)) for col in df.columns]
    columns = [re.sub(r"[,)-]", "_", str(col)) for col in columns]
    columns = [col.strip("_") for col in columns]
    # replace multiple "_" with single "_"
    columns = [re.sub(r"__+", "_", str(col)) for col in columns]
    return columns


def process_X(X, model_type):
    if model_type == "mlp":
        X = X.replace({True: 1, False: 0})
    elif model_type == "lightgbm":
        pass

    return X


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size1=128,
        hidden_size2=64,
        output_size=1,
        lr=0.001,
        batch_size=64,
        weight_decay=1e-5,
    ):
        super(MLP, self).__init__()
        # Define network layers
        # print(input_size, hidden_size1)
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.to(self.device)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier initialization for linear layers.
        """
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        # Optionally initialize biases with zeros
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation for regression task
        return x

    def fit(self, X_train, y_train, epochs=600):
        lr = self.lr
        batch_size = self.batch_size
        weight_decay = self.weight_decay

        self.train()  # Set the model in training mode
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        # Feature normalization (standardization)
        self.scaler = StandardScaler()  # Create scaler instance
        X_train_normalized = self.scaler.fit_transform(
            X_train.values
        )  # Normalize features

        # Convert pandas DataFrame/Series to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32).to(
            self.device
        )
        y_train_tensor = (
            torch.tensor(y_train.values, dtype=torch.float32)
            .unsqueeze(1)
            .to(self.device)
        )

        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in data_loader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            # 调整学习率
            scheduler.step()

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(data_loader):.6f}"
                )

    def predict(self, X_test):
        self.eval()  # Set the model in evaluation mode
        with torch.no_grad():
            X_test_normalized = self.scaler.transform(
                X_test.values
            )  # Use the same scaler as in fit
            X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32).to(
                self.device
            )
            predictions = self(X_test_tensor).cpu()
        return predictions.numpy().flatten()  # Convert predictions to numpy array


def random_search(param_dict, n_trials=10, **kwargs):
    import json

    # 存储搜索结果
    results = {}
    best_result = None
    best_mean_test_r2 = -np.inf
    best_params = None

    for _ in range(n_trials):
        params = {}
        for key, value in param_dict.items():
            if isinstance(value, list):
                params[key] = random.choice(value)  # 从列表中随机选择
            else:
                params[key] = value

        print(f"\nParams: {params}")
        _result = regressor(params=params, **kwargs)
        if _result["r2"]["test"].mean() > best_mean_test_r2:
            best_result = _result
            best_mean_test_r2 = _result["r2"]["test"].mean()
            best_params = params

        params_str = json.dumps(params, sort_keys=True)
        results[params_str] = _result

    # save best result
    df_best_params = pd.DataFrame(best_params.items(), columns=["Parameter", "Value"])
    # 添加 Search List 列，包含对应参数的搜索列表
    df_best_params["Search List"] = df_best_params["Parameter"].apply(
        lambda x: param_dict[x] if x in param_dict else None
    )

    # save best result
    best_result["best_params"] = df_best_params
    model_type = kwargs.get("model_type", "random_forest")
    suffix = kwargs.get("suffix", "2ap")
    with pd.ExcelWriter(f"./results/results_{model_type}_{suffix}.xlsx") as writer:
        for sheet_name, df in best_result.items():
            df.to_excel(writer, sheet_name=sheet_name)

    return dict(
        best_params=best_params,
        best_result=best_result,
        best_mean_test_r2=best_mean_test_r2,
    ), results


def model_selection(model_type, params, random_state=42, input_size=-1, output_size=-1):
    # 根据 model_type 选择回归模型
    if model_type == "random_forest":
        model = RandomForestRegressor(**params, random_state=random_state)
        explainer_type = "tree"
    elif model_type == "extra_trees":
        model = ExtraTreesRegressor(**params, random_state=random_state)
        explainer_type = "tree"
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            # objective="reg:squarederror",
            # n_estimators=100,
            # max_depth=3,
            # learning_rate=0.1,
            **params,
            random_state=random_state,
        )
        explainer_type = "tree"
    elif model_type == "catboost":
        model = CatBoostRegressor(
            # iterations=100,
            # learning_rate=0.1,
            # depth=3,
            **params,
        )
        explainer_type = "tree"
    elif model_type == "lightgbm":
        _params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "early_stopping_round": 10,
        }
        _params.update(params)
        model = lgb.LGBMRegressor(**_params)
        explainer_type = "tree"
    elif model_type == "mlp":
        model = MLP(**params, input_size=input_size)
        explainer_type = "deep"

    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    return model, explainer_type


def regressor(
    X, y, params, model_type="random_forest", n_splits=5, random_state=42, suffix="2ap"
):
    # 创建 KFold 交叉验证器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    # 记录训练集和测试集的指标
    metrics = dict(mse=dict(train=[], test=[]), r2=dict(train=[], test=[]))

    # 用于存储所有 SHAP 值
    all_shap_values = []

    # K 折交叉验证
    for train_index, test_index in kf.split(X):
        model, explainer_type = model_selection(
            model_type, params, random_state=random_state, input_size=X.shape[1]
        )

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train, X_test = process_X(X_train, model_type), process_X(X_test, model_type)

        # 训练模型
        if model_type == "lightgbm":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        else:
            model.fit(X_train, y_train)

        # 在训练集上进行预测并计算指标
        y_train_pred = model.predict(X_train)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        metrics["mse"]["train"].append(train_mse)
        metrics["r2"]["train"].append(train_r2)

        # 在测试集上进行预测并计算指标
        y_test_pred = model.predict(X_test)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        metrics["mse"]["test"].append(test_mse)
        metrics["r2"]["test"].append(test_r2)

        # 计算 SHAP 值
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            all_shap_values.append(np.mean(shap_values, axis=0))
        elif explainer_type == "deep":
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(
                model.device
            )
            explainer = shap.GradientExplainer(model, X_train_tensor)
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(
                model.device
            )
            shap_values = explainer.shap_values(X_test_tensor)
            all_shap_values.append(np.mean(shap_values, axis=0))

    # 输出 SHAP 值结果
    if explainer_type in ["tree", "deep"]:
        avg_shap_values = np.mean(all_shap_values, axis=0).flatten()
        feature_names = X.columns
        shap_df = pd.DataFrame(
            {"Feature": feature_names, "SHAP Value": avg_shap_values}
        )
        shap_df["SHAP Sign"] = np.sign(shap_df["SHAP Value"])
        # sort by absolute value of SHAP value
        shap_df = shap_df.reindex(
            shap_df["SHAP Value"].abs().sort_values(ascending=False).index
        )
        results["shap_values"] = shap_df

    # 如果模型有特征重要性，输出特征重要性
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        results["feature_importances"] = importance_df
        # print("\nFeature Importances:")
        # print(importance_df)

    else:
        print(f"{model_type} model does not support feature importance.")

    print(metrics)
    # turn values of metrics into dataframe
    for key, value in metrics.items():
        metrics[key] = pd.DataFrame(value)
    results.update(metrics)

    # # save results
    # with pd.ExcelWriter(f"./results/results_{model_type}_{suffix}.xlsx") as writer:
    #     for sheet_name, df in results.items():
    #         df.to_excel(writer, sheet_name=sheet_name)

    return results


class regressor_final:
    def __init__(
        self, params, model_type="random_forest", random_state=42, suffix="2ap"
    ):
        self.model_type = model_type
        self.suffix = suffix

        self.model, _ = model_selection(model_type, params, random_state=random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def plot_fit_error(self, X, y):
        y_pred = self.model.predict(X)

        # 计算均方误差
        mse = np.mean((y - y_pred) ** 2)

        # 绘制真实值 vs 预测值图
        plt.figure(figsize=(10, 6))
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")  # 45-degree line
        plt.title("MSE Distribution")
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.text(
            0.5,
            0.1,
            f"MSE: {mse:.2f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            ha="center",
        )

        # 设置坐标范围，从零开始
        plt.xlim(0, max(y.max(), y_pred.max()))
        plt.ylim(0, max(y.max(), y_pred.max()))

        # 坐标刻度一致
        plt.axis("equal")

        # 去除上、右边框
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # save
        plt.savefig(f"./fig/{self.model_type}_{self.suffix}_train_mse.svg")
        plt.show()

        # 计算误差
        errors = y - y_pred

        # 绘制误差图
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=30, alpha=0.5, color="b", edgecolor="black")
        plt.title("Error Distribution")
        plt.xlabel("Error (True - Predicted)")
        plt.ylabel("Frequency")
        plt.axvline(0, color="red", linestyle="dashed", linewidth=1)  # 0误差线
        # 去除上、右边框
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)

        # save
        plt.savefig(f"./fig/{self.model_type}_{self.suffix}_train_error.svg")

        plt.show()

    def predict(self, X):
        y_pred = self.model.predict(X)
