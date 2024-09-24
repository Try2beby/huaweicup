import re

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


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


# class MLP(BaseEstimator, RegressorMixin, nn.Module):
#     def __init__(
#         self,
#         input_size,
#         hidden_size1=128,
#         hidden_size2=64,
#         output_size=1,
#         lr=0.001,
#         batch_size=64,
#     ):
#         super(MLP, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size1)
#         self.fc2 = nn.Linear(hidden_size1, hidden_size2)
#         self.fc3 = nn.Linear(hidden_size2, output_size)
#         self.relu = nn.ReLU()

#         self.lr = lr
#         self.batch_size = batch_size
#         self.hidden_size1 = hidden_size1
#         self.hidden_size2 = hidden_size2
#         self.input_size = input_size
#         self.output_size = output_size

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.to(self.device)
#         self._initialize_weights()

#     def _initialize_weights(self):
#         nn.init.xavier_uniform_(self.fc1.weight)
#         nn.init.xavier_uniform_(self.fc2.weight)
#         nn.init.xavier_uniform_(self.fc3.weight)
#         nn.init.zeros_(self.fc1.bias)
#         nn.init.zeros_(self.fc2.bias)
#         nn.init.zeros_(self.fc3.bias)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         return self.fc3(x)

#     def fit(self, X_train, y_train, epochs=600):
#         self.scaler = StandardScaler()
#         X_train_normalized = self.scaler.fit_transform(X_train.values)

#         X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32).to(
#             self.device
#         )
#         y_train_tensor = (
#             torch.tensor(y_train.values, dtype=torch.float32)
#             .unsqueeze(1)
#             .to(self.device)
#         )

#         dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
#         data_loader = torch.utils.data.DataLoader(
#             dataset, batch_size=self.batch_size, shuffle=True
#         )

#         optimizer = optim.Adam(self.parameters(), lr=self.lr)
#         criterion = nn.MSELoss()
#         scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

#         for epoch in range(epochs):
#             running_loss = 0.0
#             for batch_X, batch_y in data_loader:
#                 optimizer.zero_grad()
#                 outputs = self(batch_X)
#                 loss = criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#                 running_loss += loss.item()

#             scheduler.step()  # Update the learning rate

#             if (epoch + 1) % 20 == 0:
#                 print(
#                     f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(data_loader):.6f}"
#                 )

#     def predict(self, X_test):
#         self.eval()
#         with torch.no_grad():
#             X_test_normalized = self.scaler.transform(X_test.values)
#             X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32).to(
#                 self.device
#             )
#             predictions = self(X_test_tensor).cpu()
#         return predictions.numpy().flatten()


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size1=128,
        hidden_size2=64,
        output_size=1,
        lr=0.001,
        batch_size=64,
    ):
        super(MLP, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

        self.lr = lr
        self.batch_size = batch_size

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

        self.train()  # Set the model in training mode
        # optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        optimizer = optim.Adam(self.parameters(), lr=lr)
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


def regressor(
    X,
    y,
    params,
    model_type="random_forest",
    scoring="r2",
    n_splits=5,
    random_state=42,
    suffix="2ap",
):
    # 根据 model_type 选择回归模型
    if model_type == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state)
        explainer_type = "tree"
    elif model_type == "xgboost":
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
        )
        explainer_type = "tree"
    elif model_type == "svr":
        model = SVR(
            kernel="rbf", C=30, epsilon=0.001
        )  # 使用 RBF 核函数，C 和 epsilon 为默认值
        explainer_type = None  # 不支持 SHAP explainer
    elif model_type == "mlp":
        model = MLP(input_size=X.shape[1])
        explainer_type = "deep"

    elif model_type == "lightgbm":
        _params = {
            "objective": "regression",
            "metric": "rmse",
            # "learning_rate": 0.1,
            # "num_leaves": 31,
            # "min_data_in_leaf": 20,
            # "feature_fraction": 0.9,
            "verbosity": -1,
            # "early_stopping_round": 10,
        }
        model = lgb.LGBMRegressor(**_params)
        explainer_type = "tree"
    elif model_type == "extra_trees":
        model = ExtraTreesRegressor(n_estimators=100, random_state=random_state)
        explainer_type = "tree"
    elif model_type == "elastic_net":
        model = ElasticNet(alpha=1e-2, l1_ratio=1e-2, max_iter=10000)
        explainer_type = None
    elif model_type == "gp":
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10.0, (1e-2, 1e2))
        model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
        explainer_type = None
    elif model_type == "catboost":
        model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3)
        explainer_type = "tree"
    else:
        raise ValueError

    # 使用 RandomizedSearchCV 进行参数搜索
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=50,
        cv=n_splits,
        # scoring="neg_mean_squared_error",
        scoring=scoring,
        random_state=random_state,
    )

    # 拟合模型
    search.fit(X, y)

    # 获取最佳模型和参数
    best_model = search.best_estimator_
    cv_results = search.cv_results_

    print(f"cv_results: {cv_results['mean_test_score']}")

    # 计算训练指标
    metrics = dict(
        r2=None,
        r2_std=None,
        mse=None,
        mse_std=None,
    )
    y_pred = best_model.predict(X)
    if scoring == "neg_mean_squared_error":
        # mean_score = cv_results["mean_test_score"].max()
        # metrics["mse"] = mean_score
        # metrics["r2"] = r2_score(y, y_pred)
        pass
    elif scoring == "r2":
        mean_test_score = cv_results["mean_test_score"].max()
        std_test_score = cv_results["std_test_score"][
            cv_results["mean_test_score"].argmax()
        ]
        metrics["r2"] = mean_test_score
        metrics["r2_std"] = std_test_score
        metrics["mse"] = mean_squared_error(y, y_pred)

    # 计算 SHAP 值
    if explainer_type == "tree":
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X)
        avg_shap_values = np.mean(shap_values, axis=0).flatten()
    elif explainer_type == "deep":
        X = process_X(X, model_type)
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(best_model.device)
        explainer = shap.GradientExplainer(best_model, X_tensor)
        shap_values = explainer.shap_values(X_tensor)
        avg_shap_values = np.mean(shap_values, axis=0).flatten()

    shap_df = None
    if explainer_type is not None:
        shap_df = pd.DataFrame({"Feature": X.columns, "SHAP Value": avg_shap_values})
        shap_df["SHAP Sign"] = np.sign(shap_df["SHAP Value"])
        shap_df = shap_df.reindex(
            shap_df["SHAP Value"].abs().sort_values(ascending=False).index
        )

    # 输出特征重要性
    importance_df = None
    if hasattr(best_model, "feature_importances_"):
        feature_importances = best_model.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": feature_importances}
        )
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

    # 获取最佳参数
    best_params = pd.DataFrame(
        search.best_params_.items(), columns=["Parameter", "Value"]
    )
    # 添加 Search List 列，包含对应参数的搜索列表
    best_params["Search List"] = best_params["Parameter"].apply(
        lambda x: params[x] if x in params else None
    )
    # 输出结果
    results = {
        "best_params": best_params,
        "metrics": pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"]),
        "shap_values": shap_df,
        "feature_importances": importance_df,
    }

    # 保存结果
    with pd.ExcelWriter(f"./results/results_{model_type}_{suffix}.xlsx") as writer:
        for sheet_name, res in results.items():
            if isinstance(res, pd.DataFrame):
                res.to_excel(writer, sheet_name=sheet_name)

    return results
