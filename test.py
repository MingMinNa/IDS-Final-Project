import pandas as pd

counties = ["南投縣", "嘉義市", "嘉義縣", "基隆市", "宜蘭縣", "屏東縣", "彰化縣", "新北市", "新竹市", "新竹縣", "桃園市", "澎湖縣", "臺中市", "臺北市", "臺南市", "臺東縣", "花蓮縣", "臺東縣", "苗栗縣", "連江縣", "金門縣", "雲林縣", "高雄市"]
file2 = pd.read_csv("data/test_data(RE).csv")

for county in counties:

    input_path = f"data/NN_AQI_result/NN_result_{county}.csv"
    file1 = pd.read_csv(input_path)


    # Merge the data based on the year, month, and day columns
    merged = pd.merge(file1, file2[['sitename', 'year', 'month', 'day', 'next_aqi']], on=['sitename', 'year', 'month', 'day'], how='left')


    output_path = f"data/NN_AQI_result(RE)/NN_result_{county}.csv"
    merged.to_csv(output_path, index=False)