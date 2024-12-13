import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def preprocess_data(X_train, X_test):
    # Separate numeric and categorical features
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    numeric_process = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

    categorical_process = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_process, numeric_features),
            ("cat", categorical_process, categorical_features),
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed


def main():
    file_path = 'data/processed/data_other_years.csv'
    train_data = pd.read_csv(file_path)
    
    file_path = 'data/processed/data_2024.csv'
    test_data = pd.read_csv(file_path)

    X_train = train_data.drop(columns=['next_aqi'])
    y_train = train_data['next_aqi']

    X_test = test_data.drop(columns=['next_aqi'])
    y_test = test_data['next_aqi']
    
    # Split the training data into training and validation sets
    tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Preprocess the data
    tmp_X_train_processed, tmp_X_val_processed = preprocess_data(tmp_X_train, tmp_X_val)
    X_train_processed, X_test_processed = preprocess_data(X_train, X_test)
    
    # Define the model
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

    # Define parameter distributions for random search
    param_distributions = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [3, 5, 7, 9, 11],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2, 0.3],
    }

    # Perform random search
    rand_search = RandomizedSearchCV(
        xgb,
        param_distributions,
        n_iter=50,
        scoring="neg_mean_squared_error",  # Regression scoring
        cv=5,  # 5-fold cross-validation
        verbose=1,
        n_jobs=-1,
        random_state=42,
    )

    # Perform random search
    rand_search.fit(tmp_X_train_processed, tmp_y_train)

    # Best model and parameters
    best_model = rand_search.best_estimator_
    best_params = rand_search.best_params_

    # print("Best Parameters:", best_params)

    # Evaluate on the validation set
    tmp_y_val_pred = best_model.predict(tmp_X_val_processed)
    mse = mean_squared_error(tmp_y_val, tmp_y_val_pred)
    r2 = r2_score(tmp_y_val, tmp_y_val_pred)

    print(f"Validation Mean Squared Error: {mse:.2f}")
    print(f"Validation R² Score: {r2:.2f}")
    # Validation Mean Squared Error: 281.19
    # Validation R² Score: 0.78

    # Save the trained model
    # best_model.save_model("best_xgb_model.json")
    # print("Model saved as 'best_xgb_model.json'")

    y_pred = best_model.predict(X_test_processed)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error of the best model: {mse}")
    # Mean Squared Error of the best model: 336.29913758809465
    
if __name__ == "__main__":
    main()