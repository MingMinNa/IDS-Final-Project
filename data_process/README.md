
### 檔案簡述
#### ``AQI_process.py``
整合 **爬蟲的AQI資料** 與 **下載的AQI資料**，大致分為以下幾步。
* ``history_AQI_filter``
    * 從下載的AQI資料中，去除爬蟲的AQI所沒有的測站資料。  
    例如：在爬蟲AQI中，並沒有**關山**測站相關資料，則將其測站資料從下載AQI移除。
    * 去除下載AQI的``datacreationdate``特徵中的小時與分鐘。  
    例如：將 ``2024-01-01 00:00`` 轉換成 ``2024-01-01``
    * 將轉換過的下載AQI，儲存到 ``ming`` 資料夾中的 ``tmp`` 資料夾。
* ``generate_result``
    * 基於爬蟲AQI的日期與測站，從轉換過的AQI中，找到日期與測站相同的資料，並計算平均，作為該測站當天的汙染物平均。即計算當天不同小時汙染物的平均。
    * 最終結果儲存在 ``ming`` 資料夾內的 ``result.csv``。

* ``result_handle``
    * 基於日期與測站，對 ``result.csv`` 進行排序。
    * 將資料做四捨五入。

* ``add_next_aqi``
    * 替 ``final_result.csv`` 加上隔日的AQI（``next_aqi``）標籤，作為訓練時的輸出特徵使用。

註：
1. 在 ``result.csv`` 資料中，``aqi``標籤為利用平均所計算出來的，而 ``aqi_2`` 則是從爬蟲AQI中取得的。
2. 在 ``result.csv`` 資料中，有部分資料缺失，是由於原先下載AQI便沒有該筆資料，例如：``2021-07-30`` 以及 ``2021-08-01``等，這部分還請注意。
3. ``2021-05-01`` 至 ``2021-05-31`` 在爬蟲AQI中並沒有資料，因此，將以 ``aqi`` 數值補充 ``aqi_2``

#### ``rainfall_process.py``
在 ``AQI_process.py`` 生成 ``result.csv`` 後，將**降雨量資料**加進 ``result.csv``，並重新命名為 ``final_result.csv``（後將 ``final_result.csv`` 重新命名為 ``train_data.csv``）。

#### ``test_process.py``
利用下載資料中的 ``2024-10-01`` 至 ``2024-11-30`` 資料，製作測試資料集 ``test_data.csv``，放置於 ``ming`` 資料夾內。 
