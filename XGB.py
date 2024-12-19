import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

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

    return X_train_processed, X_test_processed, preprocessor


def train_predict_model(train_data, test_data, target_column, target_columns):
    # Prepare features and target
    X_train = train_data.drop(columns=target_columns)
    X_train = train_data.drop(columns=["county"])
    y_train = train_data[target_column]

    X_test = test_data.drop(columns=target_columns)
    X_test = test_data.drop(columns=["county"])
    y_test = test_data[target_column]
    
    # Split the training data into training and validation sets
    tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42)

    # Preprocess the data
    tmp_X_train_processed, tmp_X_val_processed, preprocessor = preprocess_data(tmp_X_train, tmp_X_val)
    X_train_processed, X_test_processed, _ = preprocess_data(X_train, X_test)
    
    # Define the model
    xgb = XGBRegressor(
        verbosity=1,               # Optional: set verbosity to see training info
        objective='reg:absoluteerror',
        random_state=42
    )
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
        scoring='neg_mean_absolute_error',
        cv=5,
        verbose=0,
        n_jobs=-1,
        random_state=42,
    )

    # Fit the model
    rand_search.fit(tmp_X_train_processed, tmp_y_train)

    # Best model
    best_model = rand_search.best_estimator_

    # Evaluate on the validation set
    tmp_y_val_pred = best_model.predict(tmp_X_val_processed)
    mae = mean_absolute_error(tmp_y_val, tmp_y_val_pred)

    print(f"Model for {target_column}:")
    print(f"Validation Mean Absolute Error: {mae:.2f}")

    # Predict on test data
    y_pred = best_model.predict(X_test_processed)

    # Create a result DataFrame
    result_df = test_data[['year', 'month', 'day', 'sitename']].copy()
    result_df[f'predicted_{target_column}'] = y_pred
    result_df[f'actual_{target_column}'] = y_test.values

    return result_df, mae


def main():
    # Read the data
    train_data = pd.read_csv('data/train_data.csv')
    test_data = pd.read_csv('data/test_data.csv')

    # List of target columns to predict
    target_columns = [
        'next_so2', 'next_co', 'next_o3_8hr', 
        'next_o3', 'next_pm2.5', 'next_pm10', 'next_no2'
    ]

    # target_columns = ['next_so2'] # temp

    # Store all results
    all_results = []
    performance_metrics = {}

    unique_counties = train_data['county'].unique()

    for county in unique_counties:
        print(f"Processing data for county: {county}")
        current_train_data = train_data[train_data['county'] == county]
        current_test_data = test_data[test_data['county'] == county]

        # Initialize a DataFrame to store all results for the current county
        county_results = pd.DataFrame()

        # Train and predict for each target column
        for target_column in target_columns:
            result_df, mae= train_predict_model(current_train_data, current_test_data, target_column, target_columns)

            # Add the target column name as a prefix to the result columns to avoid conflicts
            # result_df = result_df.add_prefix(f"{target_column}_")

            # Concatenate the result with the county_results DataFrame
            if county_results.empty:
                county_results = result_df
            else:
                county_results = pd.concat([county_results, result_df], axis=1)

        # Add the county column to the county_results DataFrame
        county_results['county'] = county

        # Save the aggregated results to a CSV file
        filename = f"data/XGB_result_{county}.csv"
        county_results.to_csv(filename, index=False)


if __name__ == "__main__":
    main()