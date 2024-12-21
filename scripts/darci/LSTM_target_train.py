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

project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
train_data_path = os.path.join(project_root, "data", "darci", "train_data.csv")
test_data_path = os.path.join(project_root, "data", "darci", "test_data.csv")

model_dir = os.path.join(project_root, "models", "darci")
os.makedirs(model_dir, exist_ok=True)

img_dir = os.path.join(project_root, "img", "LSTM")
os.makedirs(img_dir, exist_ok=True)


def create_sequences(data, labels, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(data) - seq_length):
        X_seq.append(data[i : i + seq_length])
        y_seq.append(labels[i + seq_length])
    return np.array(X_seq), np.array(y_seq)


def main():
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)

    features = [
        "aqi_2",
        "so2",
        "co",
        "o3",
        "o3_8hr",
        "pm10",
        "pm2.5",
        "no2",
        "nox",
        "no",
        "windspeed,winddirec",
        "co_8hr",
        "pm2.5_avg",
        "pm10_avg",
        "next_precipitation",
    ]

    target = "next_aqi"

    seq_length = 15

    site_models = {}
    for sitename, site_train_data in train_data.groupby("sitename"):
        print(f"Processing site: {sitename}")

        site_train_data = site_train_data.sort_values(["year", "month", "day"])
        site_test_data = test_data[test_data["sitename"] == sitename].sort_values(["year", "month", "day"])

        X_train = site_train_data[features].values
        y_train = site_train_data[target].values
        X_test = site_test_data[features].values
        y_test = site_test_data[target].values

        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
        X_test_scaled = scaler_X.transform(X_test)
        y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

        X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, seq_length)
        X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, seq_length)

        model = Sequential(
            [
                LSTM(128, input_shape=(seq_length, len(features)), return_sequences=True),
                LSTM(64, return_sequences=True),
                LSTM(32),
                Dense(1),
            ]
        )

        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        history = model.fit(
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
        scaler_X_path = os.path.join(model_dir, "LSTM_scaler", f"{sitename}_scaler_X.pkl")
        scaler_y_path = os.path.join(model_dir, "LSTM_scaler", f"{sitename}_scaler_y.pkl")

        save_model(model, model_path)
        joblib.dump(scaler_X, scaler_X_path)
        joblib.dump(scaler_y, scaler_y_path)

        site_models[sitename] = {
            "model": model,
            "scaler_X": scaler_X,
            "scaler_y": scaler_y,
            "X_test_seq": X_test_seq,
            "y_test_seq": y_test_seq,
            "history": history,
        }

    for sitename, model_data in site_models.items():
        model = model_data["model"]
        X_test_seq = model_data["X_test_seq"]
        y_test_seq = model_data["y_test_seq"]
        scaler_y = model_data["scaler_y"]
        history = model_data["history"]

        y_pred_seq = model.predict(X_test_seq)

        y_test_inverse = scaler_y.inverse_transform(y_test_seq)
        y_pred_inverse = scaler_y.inverse_transform(y_pred_seq)

        # 繪製訓練過程中的 LOSS 與 MAE 曲線
        train_loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        train_mae = history.history["mae"]
        val_mae = history.history["val_mae"]

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.plot(train_mae, label="Train MAE")
        plt.plot(val_mae, label="Val MAE")
        plt.title(f"{sitename} Training & Validation Loss/MAE")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plot_metric_path = os.path.join(img_dir, "LSTM_train", f"{sitename}_training_metrics_plot.png")
        plt.savefig(plot_metric_path)
        plt.close()
        print(f"Training metric plot saved for site {sitename} at {plot_metric_path}")

        # 繪製 Actual vs Predicted 圖
        plt.figure(figsize=(10, 6))
        plt.plot(y_test_inverse, label="Actual")
        plt.plot(y_pred_inverse, label="Predicted")
        plt.title(f"Site: {sitename} - Actual vs Predicted AQI")
        plt.xlabel("Time Steps")
        plt.ylabel("AQI")
        plt.legend()
        plt.grid(True)

        plot_path = os.path.join(img_dir, "LSTM_result", f"{sitename}_prediction_plot.png")
        plt.savefig(plot_path)
        plt.close()

        print(f"Prediction plot saved for site {sitename} at {plot_path}")


if __name__ == "__main__":
    main()
