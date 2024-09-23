import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def process_X(X):
    X = X.replace({True: 1, False: 0})
    return X


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1=128, hidden_size2=64, output_size=1):
        super(MLP, self).__init__()
        # Define network layers
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

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

    def fit(self, X_train, y_train, epochs=600, lr=0.001, batch_size=64):
        self.train()  # Set the model in training mode
        # optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        # Feature normalization (standardization)
        self.scaler = StandardScaler()  # Create scaler instance
        X_train = process_X(X_train)  # Replace boolean with 1/0
        X_train_normalized = self.scaler.fit_transform(
            X_train.values
        )  # Normalize features

        # Convert pandas DataFrame/Series to PyTorch tensors
        X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)

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
            X_test = process_X(X_test)
            X_test_normalized = self.scaler.transform(
                X_test.values
            )  # Use the same scaler as in fit
            X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
            predictions = self(X_test_tensor)
        return predictions.numpy().flatten()  # Convert predictions to numpy array


def regressor(X, y, model_type="random_forest", n_splits=5, random_state=42):
    """
    参数:
    - X: 特征数据集 (pandas DataFrame)
    - y: 目标数据集 (pandas Series)
    - model_type: 选择使用的回归模型，'random_forest'、'xgboost' 或 'svr'
    - n_splits: K折交叉验证的折数
    - random_state: 随机种子，确保结果可复现
    """
    # 创建 KFold 交叉验证器
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

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
    else:
        raise ValueError

    mse_scores = []
    r2_scores = []

    # 用于存储所有 SHAP 值
    all_shap_values = []

    # K 折交叉验证
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # 训练模型
        model.fit(X_train, y_train)

        # 进行预测
        y_pred = model.predict(X_test)

        # 评估模型
        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

        # 计算 SHAP 值
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            all_shap_values.append(np.mean(shap_values, axis=0))
        elif explainer_type == "deep":
            X_train = process_X(X_train)
            X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
            explainer = shap.GradientExplainer(model, X_train_tensor)
            X_test = process_X(X_test)
            X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
            shap_values = explainer.shap_values(X_test_tensor)
            all_shap_values.append(np.mean(shap_values, axis=0))

    # 输出交叉验证结果
    print(f"Mean Squared Error (MSE) scores: {mse_scores}")
    print(f"Average MSE: {np.mean(mse_scores)}")
    print(f"R^2 scores: {r2_scores}")
    print(f"Average R^2: {np.mean(r2_scores)}")

    # 输出 SHAP 值结果
    if explainer_type in ["tree", "deep"]:
        avg_shap_values = np.mean(all_shap_values, axis=0).flatten()
        feature_names = X.columns
        shap_df = pd.DataFrame(
            {"Feature": feature_names, "SHAP Value": avg_shap_values}
        )
        shap_df = shap_df.sort_values(by="SHAP Value", ascending=False)
        results["shap_values"] = shap_df
        print("\nSHAP Values:")
        print(shap_df)

    # 如果模型有特征重要性，输出特征重要性
    if hasattr(model, "feature_importances_"):
        feature_importances = model.feature_importances_
        feature_names = X.columns
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": feature_importances}
        )
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        # 验证特征重要性之和为 1
        assert np.isclose(importance_df.Importance.sum(), 1.0)
        results["feature_importances"] = importance_df
        print("\nFeature Importances:")
        print(importance_df)

    else:
        print(f"{model_type} model does not support feature importance.")

    return results
