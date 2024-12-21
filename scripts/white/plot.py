import pandas as pd
import matplotlib.pyplot as plt
import os


counties = ["南投縣", "嘉義市", "嘉義縣", "基隆市", "宜蘭縣", "屏東縣", "彰化縣", "新北市", "新竹市", "新竹縣", "桃園市", "澎湖縣", "臺中市", "臺北市", "臺南市", "臺東縣", "花蓮縣", "臺東縣", "苗栗縣", "連江縣", "金門縣", "雲林縣", "高雄市"]

for county in counties:
    # 讀取 CSV 文件
    PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    input_path = os.path.join(PROJECT_FOLDER, 'data', 'white', 'XGB_AQI_result(RE)', f"XGB_result_{county}.csv")
    output_path = os.path.join(PROJECT_FOLDER, 'data', 'white', 'NN_AQI_result(RE)', f"NN_result_{county}.csv")

    file = pd.read_csv(input_path)

    # 確保日期順序
    file = file.sort_values(by=['sitename', 'year', 'month', 'day'])

    # 取得所有唯一的 sitename
    sitename_list = file['sitename'].unique()

    # 確保存放圖片的目錄存在
    os.makedirs('img', exist_ok=True)

    # 繪製每個 sitename 的折線圖並保存為圖片
    for sitename in sitename_list:
        subset = file[file['sitename'] == sitename]
        # 將日期格式化為 YYYY-MM-DD
        dates = subset.apply(lambda row: f"{int(row['year'])}-{int(row['month']):02d}-{int(row['day']):02d}", axis=1)
        
        plt.figure(figsize=(10, 6))
        plt.plot(dates, subset['next_aqi'], label='Actual AQI', marker='x')
        plt.plot(dates, subset['predicted_AQI'], label='Predicted AQI', marker='o')

        plt.title(f'AQI Trends for {sitename}', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('AQI', fontsize=12)
        plt.legend()
        plt.grid(True)

        # 設置 X 軸顯示每五天的日期
        xticks_positions = range(0, len(dates), 5)  # 每隔五天的位置
        xticks_labels = [dates.iloc[i] for i in xticks_positions]  # 對應的日期
        plt.xticks(xticks_positions, xticks_labels, rotation=45)

        plt.tight_layout()
        
        # 儲存圖片
        filename = os.path.join(PROJECT_FOLDER, 'img', 'white', 'NN', f"{county}_{sitename}_prediction_plot.png")
        plt.savefig(filename)
        plt.close()  # 關閉圖表，釋放內存
