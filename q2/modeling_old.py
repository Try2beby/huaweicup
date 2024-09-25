import random
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from catboost import CatBoostClassifier
from matplotlib_inline import backend_inline
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

backend_inline.set_matplotlib_formats("svg")


def clear_column_names(df):
    columns = [re.sub(r"[\'\"(\s]", "", str(col)) for col in df.columns]
    columns = [re.sub(r"[,)-]", "_", str(col)) for col in columns]
    columns = [col.strip("_") for col in columns]
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
        output_size=1,  # Change this to number of classes for multi-class classification
        lr=0.001,
        batch_size=64,
        weight_decay=1e-5,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(
            hidden_size2, output_size
        )  # Output layer for classification
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(
            dim=1
        )  # Softmax activation for multi-class classification

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # No activation here, applied later
        return self.softmax(x)  # Apply softmax activation

    def fit(self, X_train, y_train, epochs=600):
        lr = self.lr
        batch_size = self.batch_size
        weight_decay = self.weight_decay

        self.train()
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

        self.scaler = StandardScaler()
        X_train_normalized = self.scaler.fit_transform(X_train.values)

        X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32).to(
            self.device
        )
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.long).to(
            self.device
        )  # Use long for class labels

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

            scheduler.step()

            if (epoch + 1) % 20 == 0:
                print(
                    f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(data_loader):.6f}"
                )

    def predict(self, X_test):
        self.eval()
        with torch.no_grad():
            X_test_normalized = self.scaler.transform(X_test.values)
            X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32).to(
                self.device
            )
            outputs = self(X_test_tensor)
            _, predicted = torch.max(
                outputs, 1
            )  # Get the index of the max log-probability
        return predicted.cpu().numpy()  # Convert predictions to numpy array


def random_search(param_dict, n_trials=10, **kwargs):
    import json

    results = {}
    best_result = None
    best_mean_test_f1 = -np.inf
    best_params = None

    for _ in range(n_trials):
        params = {
            key: random.choice(value) if isinstance(value, list) else value
            for key, value in param_dict.items()
        }

        print(f"\nParams: {params}")
        _result = classifier(params=params, **kwargs)  # Change to classifier
        if _result["f1"]["test"].mean() > best_mean_test_f1:
            best_result = _result
            best_mean_test_f1 = _result["f1"]["test"].mean()
            best_params = params

        results[json.dumps(params, sort_keys=True)] = _result

    df_best_params = pd.DataFrame(best_params.items(), columns=["Parameter", "Value"])
    df_best_params["Search List"] = df_best_params["Parameter"].apply(
        lambda x: param_dict[x] if x in param_dict else None
    )

    best_result["best_params"] = df_best_params
    model_type = kwargs.get("model_type", "random_forest")
    suffix = kwargs.get("suffix", "2ap")
    with pd.ExcelWriter(f"./results/results_{model_type}_{suffix}.xlsx") as writer:
        for sheet_name, df in best_result.items():
            df.to_excel(writer, sheet_name=sheet_name)

    return dict(
        best_params=best_params,
        best_result=best_result,
        best_mean_test_f1=best_mean_test_f1,
    ), results


def model_selection(model_type, params, random_state=42, input_size=-1, output_size=-1):
    if model_type == "random_forest":
        model = RandomForestClassifier(**params, random_state=random_state)
        explainer_type = "tree"
    elif model_type == "extra_trees":
        model = ExtraTreesClassifier(**params, random_state=random_state)
        explainer_type = "tree"
    elif model_type == "xgboost":
        model = xgb.XGBClassifier(**params, random_state=random_state)
        explainer_type = "tree"
    elif model_type == "catboost":
        model = CatBoostClassifier(**params)
        explainer_type = "tree"
    elif model_type == "lightgbm":
        _params = {
            "objective": "multiclass",
            "metric": "multi_logloss",
            "learning_rate": 0.1,
            "num_leaves": 31,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "early_stopping_round": 10,
        }
        _params.update(params)
        model = lgb.LGBMClassifier(**_params)
        explainer_type = "tree"
    elif model_type == "mlp":
        model = MLP(
            **params, input_size=input_size
        )  # Set output size to number of classes
        explainer_type = "deep"

    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    return model, explainer_type


def classifier(
    X, y, params, model_type="random_forest", n_splits=5, random_state=42, suffix="2ap"
):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    results = {}

    metrics = dict(accuracy=dict(train=[], test=[]), f1=dict(train=[], test=[]))

    # 用于存储所有 SHAP 值
    all_shap_values = []

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

        y_train_pred = model.predict(X_train)
        metrics["accuracy"]["train"].append(accuracy_score(y_train, y_train_pred))
        metrics["f1"]["train"].append(
            f1_score(y_train, y_train_pred, average="weighted")
        )

        y_test_pred = model.predict(X_test)
        metrics["accuracy"]["test"].append(accuracy_score(y_test, y_test_pred))
        metrics["f1"]["test"].append(f1_score(y_test, y_test_pred, average="weighted"))

        # 计算 SHAP 值
        if explainer_type == "tree":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            # print(X_test.shape, shap_values.shape)
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

    # # 输出 SHAP 值结果
    # if explainer_type in ["tree", "deep"]:
    #     avg_shap_values = np.mean(all_shap_values, axis=0).flatten()
    #     feature_names = X.columns
    #     # print(len(avg_shap_values), len(feature_names))
    #     shap_df = pd.DataFrame(
    #         {"Feature": feature_names, "SHAP Value": avg_shap_values}
    #     )
    #     shap_df["SHAP Sign"] = np.sign(shap_df["SHAP Value"])
    #     # sort by absolute value of SHAP value
    #     shap_df = shap_df.reindex(
    #         shap_df["SHAP Value"].abs().sort_values(ascending=False).index
    #     )
    #     results["shap_values"] = shap_df

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
