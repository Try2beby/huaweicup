import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


def random_forest_regressor(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    # 输出交叉验证结果
    print(f"Mean Squared Error (MSE) scores: {mse_scores}")
    print(f"Average MSE: {np.mean(mse_scores)}")
    print(f"R^2 scores: {r2_scores}")
    print(f"Average R^2: {np.mean(r2_scores)}")

    # 输出特征重要性
    feature_importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": feature_importances}
    )
    importance_df = importance_df.sort_values(by="Importance", ascending=False)

    print("\nFeature Importances:")
    print(importance_df)
