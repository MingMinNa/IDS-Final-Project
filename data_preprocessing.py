import pandas as pd
import os


project_root = os.getcwd()
train_data_path = os.path.join(project_root, "data", "train_data.csv")
test_data_path = os.path.join(project_root, "data", "test_data.csv")


def add_precipitation_next_column(file_path):
    df = pd.read_csv(file_path)

    if "precipitation" in df.columns:
        df["next_precipitation"] = df["precipitation"].shift(-1)

    if "datacreationdate" in df.columns:
        df["datacreationdate"] = pd.to_datetime(df["datacreationdate"])
        df["year"] = df["datacreationdate"].dt.year
        df["month"] = df["datacreationdate"].dt.month
        df["day"] = df["datacreationdate"].dt.day

    df = df.dropna()
    df.to_csv(file_path, index=False)
    print(f"File saved to {file_path}")


def main():
    add_precipitation_next_column(train_data_path)
    add_precipitation_next_column(test_data_path)


if __name__ == "__main__":
    main()
