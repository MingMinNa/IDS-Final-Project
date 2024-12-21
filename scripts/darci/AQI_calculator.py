import pandas as pd
import os
import matplotlib.pyplot as plt

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')
IMG_FOLDER = os.path.join(PROJECT_FOLDER, 'img')

input_csv = os.path.join(DATA_FOLDER, "darci", "result", "LSTM_pred_with_precipitation.csv")
output_csv = os.path.join(DATA_FOLDER, "darci", "result", "LSTM_calc.csv")

img_dir = os.path.join(IMG_FOLDER, "darci", "LSTM_pred_features_with_precipitation")
os.makedirs(img_dir, exist_ok=True)


def calc_sub_index(pollutant, value):
    breakpoints = {
        "o3_8hr": [
            (0.000, 0.054, 0, 50),
            (0.055, 0.070, 51, 100),
            (0.071, 0.085, 101, 150),
            (0.086, 0.105, 151, 200),
            (0.106, 0.200, 201, 300),
        ],
        "o3_1hr": [
            (0.125, 0.164, 101, 150),
            (0.165, 0.204, 151, 200),
            (0.205, 0.404, 201, 300),
            (0.405, 0.504, 301, 400),
            (0.505, 0.604, 401, 500),
        ],
        "pm2.5": [
            (0.0, 15.4, 0, 50),
            (15.5, 35.4, 51, 100),
            (35.5, 54.4, 101, 150),
            (54.5, 150.4, 151, 200),
            (150.5, 250.4, 201, 300),
            (250.5, 350.4, 301, 400),
            (350.5, 500.4, 401, 500),
        ],
        "pm10": [
            (0, 50, 0, 50),
            (50, 100, 51, 100),
            (101, 254, 101, 150),
            (255, 354, 151, 200),
            (355, 424, 201, 300),
            (425, 504, 301, 400),
            (505, 604, 401, 500),
        ],
        "co": [
            (0.0, 4.4, 0, 50),
            (4.5, 9.4, 51, 100),
            (9.5, 12.4, 101, 150),
            (12.5, 15.4, 151, 200),
            (15.5, 30.4, 201, 300),
            (30.5, 40.4, 301, 400),
            (40.5, 50.4, 401, 500),
        ],
        "so2": [
            (0, 20, 0, 50),
            (21, 75, 51, 100),
            (76, 185, 101, 150),
            (186, 304, 151, 200),
            (305, 604, 201, 300),
            (605, 804, 301, 400),
            (805, 1004, 401, 500),
        ],
        "no2": [
            (0, 30, 0, 50),
            (31, 100, 51, 100),
            (101, 360, 101, 150),
            (361, 649, 151, 200),
            (650, 1249, 201, 300),
            (1250, 1649, 301, 400),
            (1650, 2049, 401, 500),
        ],
    }

    if pollutant not in breakpoints or value is None:
        return 0

    bp_list = breakpoints[pollutant]
    for BP_L, BP_H, I_L, I_H in bp_list:
        if BP_L <= value <= BP_H:
            I = (I_H - I_L) / (BP_H - BP_L) * (value - BP_L) + I_L
            return I

    return 0


def compute_aqi(row):
    so2 = row["pred_so2"]
    co = row["pred_co"]
    o3_8hr = row["pred_o3_8hr"]
    o3_1hr = row["pred_o3"]
    pm10 = row["pred_pm10"]
    pm25 = row["pred_pm2.5"]
    no2 = row["pred_no2"]

    o3_8hr_i = calc_sub_index("o3_8hr", o3_8hr)
    o3_1hr_i = calc_sub_index("o3_1hr", o3_1hr)
    o3_i = max(o3_8hr_i, o3_1hr_i)

    pm25_i = calc_sub_index("pm2.5", pm25)
    pm10_i = calc_sub_index("pm10", pm10)
    co_i = calc_sub_index("co", co)
    so2_i = calc_sub_index("so2", so2)
    no2_i = calc_sub_index("no2", no2)

    aqi = max(o3_i, pm25_i, pm10_i, co_i, so2_i, no2_i)
    return aqi


def cm_to_inch(cm):
    return cm / 2.54


def main():
    df = pd.read_csv(input_csv)
    df["calc_AQI"] = df.apply(compute_aqi, axis=1)
    df["calc_AQI"] = df["calc_AQI"].round(3)
    df["abs_diff"] = (df["aqi_2"] - df["calc_AQI"]).abs()

    total_abs_diff = df["abs_diff"].sum()
    print(f"Total Absolute Difference between aqi_2 and calc_AQI: {total_abs_diff}")

    df = df[["datacreationdate", "sitename", "aqi_2", "calc_AQI", "abs_diff"]]

    df.to_csv(output_csv, index=False)
    print("File saved at ", output_csv)

    df["datacreationdate"] = pd.to_datetime(df["datacreationdate"], errors="coerce")

    for sitename, group in df.groupby("sitename"):
        group = group.sort_values("datacreationdate")
        plt.figure(figsize=(10, 6))
        plt.plot(group["datacreationdate"], group["aqi_2"], marker="x", label="Actual AQI")
        plt.plot(group["datacreationdate"], group["calc_AQI"], marker="o", label="Predicted AQI")

        plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.title(f"AQI Comparison for {sitename}")
        plt.xlabel("Date")
        plt.ylabel("AQI")
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.legend()

        plot_path = os.path.join(img_dir, f"{sitename}_AQI_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved for site {sitename} at {plot_path}")


if __name__ == "__main__":
    main()
