from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense  # type: ignore
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from tensorflow.keras.saving import save_model  # type: ignore

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os

project_root = os.getcwd()
train_data_path = os.path.join(project_root, "data", "train_data.csv")
test_data_path = os.path.join(project_root, "data", "test_data.csv")

model_dir = os.path.join(project_root, "model")
os.makedirs(model_dir, exist_ok=True)

output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)

img_dir = os.path.join(project_root, "img")
os.makedirs(img_dir, exist_ok=True)


def create_sequences(data, seq_length):
    X_seq = []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i : i + seq_length])
    return np.array(X_seq)


def main():
    features = [
        "so2",
        "co",
        "o3",
        "o3_8hr",
        "pm10",
        "pm2.5",
        "no2",
        "nox",
        "no",
        "windspeed",
        "winddirec",
        "co_8hr",
        "pm2.5_avg",
        "pm10_avg",
    ]

    seq_length = 15

    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    all_sites_results = []

    for sitename, site_train_data in train_data.groupby("sitename"):
        print(f"Processing site: {sitename}")

        site_train_data = site_train_data.sort_values(["year", "month", "day"])
        site_test_data = test_data[test_data["sitename"] == sitename].sort_values(["year", "month", "day"])

        X_train = site_train_data[features].values
        X_test = site_test_data[features].values

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_seq = create_sequences(X_train_scaled, seq_length)
        X_test_seq = create_sequences(X_test_scaled, seq_length)

        y_train_seq = X_train_scaled[seq_length:]
        y_test_seq = X_test_scaled[seq_length:]

        model = Sequential(
            [
                LSTM(128, input_shape=(seq_length, len(features)), return_sequences=True),
                LSTM(64, return_sequences=True),
                LSTM(32),
                Dense(len(features)),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

        model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_test_seq, y_test_seq),
            epochs=50,
            batch_size=32,
            callbacks=[early_stopping],
        )

        loss, mae = model.evaluate(X_test_seq, y_test_seq, verbose=0)
        print(f"Site {sitename} Test Loss: {loss}, Test MAE: {mae}")

        model_path = os.path.join(model_dir, "LSTM_model", f"{sitename}_lstm_model.keras")
        scaler_path = os.path.join(model_dir, "LSTM_scaler", f"{sitename}_scaler.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

        save_model(model, model_path)
        joblib.dump(scaler, scaler_path)

        y_pred_seq = model.predict(X_test_seq)

        y_pred_inverse = scaler.inverse_transform(y_pred_seq)

        prediction_dates = site_test_data.iloc[seq_length:][["datacreationdate", "sitename", "aqi_2"]].copy()
        pred_df = pd.DataFrame(y_pred_inverse, columns=[f"pred_{f}" for f in features])
        result_df = pd.concat([prediction_dates.reset_index(drop=True), pred_df], axis=1)

        all_sites_results.append(result_df)

    final_result_df = pd.concat(all_sites_results, ignore_index=True)
    final_csv_path = os.path.join(output_dir, "LSTM_predictions.csv")
    final_result_df.to_csv(final_csv_path, index=False)
    print(f"All sites predictions saved at {final_csv_path}")


if __name__ == "__main__":
    main()
