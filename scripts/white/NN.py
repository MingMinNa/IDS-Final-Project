import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
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


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def train_predict_model(train_data, test_data, target_column, target_columns):
    # Prepare features and target
    X_train = train_data.drop(columns=target_columns)
    X_train = X_train.drop(columns=["county"])
    y_train = train_data[target_column]

    X_test = test_data.drop(columns=target_columns)
    X_test = X_test.drop(columns=["county"])
    y_test = test_data[target_column]

    # Split the training data into training and validation sets
    tmp_X_train, tmp_X_val, tmp_y_train, tmp_y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    # Preprocess the data
    tmp_X_train_processed, tmp_X_val_processed, preprocessor = preprocess_data(tmp_X_train, tmp_X_val)
    X_train_processed, X_test_processed, _ = preprocess_data(X_train, X_test)

    # Convert data to PyTorch tensors
    tmp_X_train_tensor = torch.tensor(tmp_X_train_processed, dtype=torch.float32)
    tmp_y_train_tensor = torch.tensor(tmp_y_train.values, dtype=torch.float32).view(-1, 1)
    tmp_X_val_tensor = torch.tensor(tmp_X_val_processed, dtype=torch.float32)
    tmp_y_val_tensor = torch.tensor(tmp_y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_processed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Define a simple feedforward neural network
    class RegressionModel(nn.Module):
        def __init__(self, input_dim):
            super(RegressionModel, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )

        def forward(self, x):
            return self.fc(x)

    # Initialize the model, loss function, and optimizer
    input_dim = tmp_X_train_tensor.shape[1]
    model = RegressionModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    batch_size = 32
    train_dataset = TensorDataset(tmp_X_train_tensor, tmp_y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_predictions = model(tmp_X_val_tensor)
            val_loss = criterion(val_predictions, tmp_y_val_tensor).item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
        mae = mean_absolute_error(y_test_tensor.numpy(), y_pred)

    print(f"Model for {target_column}: Validation Mean Absolute Error: {mae:.2f}")

    # Create a result DataFrame
    result_df = test_data[["year", "month", "day", "sitename"]].copy()
    result_df[f"predicted_{target_column}"] = y_pred
    result_df[f"actual_{target_column}"] = y_test.values

    return result_df, mae



def main():
    # Read the data
    PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    train_data = pd.read_csv(os.path.join(PROJECT_FOLDER, 'data', 'white', 'train_data.csv'))
    test_data = pd.read_csv(os.path.join(PROJECT_FOLDER, 'data', 'white', 'test_data.csv'))

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
        filename = os.path.join(PROJECT_FOLDER, 'data', 'white', 'NN_AQI_result', f"NN_result_{county}.csv")
        county_results.to_csv(filename, index=False)


if __name__ == "__main__":
    main()