import os 
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
RAW_FOLDER = os.path.join(PROJECT_FOLDER, 'data', 'raw')
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, 'data', 'ming')

aqi_sites = ['萬里', '安南', '冬山', '忠明', '新港', '大寮', '士林', '臺南', 
             '臺東', '板橋', '豐原', '龍潭', '大里', '淡水', '土城', '宜蘭', 
             '中山', '左營', '仁武', '屏東', '林口', '林園', '前金', '大園', 
             '南投', '馬祖', '湖口', '桃園', '嘉義', '潮州', '新營', '花蓮', 
             '新竹', '彰化', '苗栗', '金門', '西屯', '美濃', '馬公', '基隆', 
             '恆春', '新莊', '善化', '二林', '小港', '松山', '竹山', '平鎮', 
             '朴子', '楠梓', '三義', '崙背', '古亭', '沙鹿', '汐止', '斗六', 
             '新店', '菜寮', '竹東', '萬華']
site_mapping =  ['NewTaipei', 'Tainan', 'Yilan', 'Taichung', 'Chiayi', 'Kaohsiung', 'Taipei', 'Tainan', 
                 'Taitung', 'NewTaipei', 'Taichung', 'Taoyuan', 'Taichung', 'NewTaipei', 'NewTaipei', 'Yilan', 
                 'Taipei', 'Kaohsiung', 'Kaohsiung', 'Pingtung', 'NewTaipei', 'Kaohsiung', 'Kaohsiung', 'Taoyuan', 
                 'SunMoonLake', 'Matsu', 'Hsinchu', 'Taoyuan', 'Chiayi', 'Pingtung', 'Tainan', 'Hualian', 
                 'Hsinchu', 'Changhua', 'Miaoli', 'Kinmen', 'Taichung', 'Kaohsiung', 'Penghu', 'Keelung', 
                 'Pingtung', 'NewTaipei', 'Tainan', 'Changhua', 'Kaohsiung', 'Taipei', 'SunMoonLake', 'Taoyuan', 
                 'Chiayi', 'Kaohsiung', 'Miaoli', 'Yunlin', 'Taipei', 'Taichung', 'NewTaipei', 'Yunlin', 
                 'NewTaipei', 'NewTaipei', 'Hsinchu', 'Taipei']
aqi_sites = {aqi:rainfall for aqi, rainfall in zip(aqi_sites, site_mapping)}

rainfall_sites = [
    'Changhua',     # 彰化
    'Chiayi',       # 嘉義
    'Hsinchu',      # 新竹
    'Hualian',      # 花蓮
    'Kaohsiung',    # 高雄
    'Keelung',      # 基隆
    'Miaoli',       # 苗栗
    'NewTaipei',    # 新北
    'Pingtung',     # 屏東
    'Taichung',     # 台中
    'Tainan',       # 台南
    'Taipei',       # 台北
    'Taitung',      # 台東
    'Taoyuan',      # 桃園
    'Yilan',        # 宜蘭
    'Yunlin',       # 雲林
    'Kinmen',       # 金門
    'Matsu',        # 馬祖
    'Penghu',       # 澎湖
    'SunMoonLake',  # 日月潭（南投）
]


if __name__ == '__main__':
    test_aqi_df = pd.concat([
        pd.read_csv(os.path.join(RAW_FOLDER, 'AQI', f'空氣品質指標(AQI)(歷史資料) (2024-{month:02d}).csv')) for month in [10, 11]
    ], axis = 0)

    raw_AQI_features = 'sitename,county,aqi,pollutant,status,so2,co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,windspeed,winddirec,datacreationdate,unit,co_8hr,pm2.5_avg,pm10_avg,so2_avg,longitude,latitude,siteid'.split(',')

    # most of 'so2_avg' is NaN, don't treat this feature as an input feature
    removed_features = 'status,longitude,latitude,siteid,pollutant,unit,so2_avg'.split(',')
    remaining_features = [feature for feature in raw_AQI_features if feature not in removed_features]

    test_aqi_df = test_aqi_df[test_aqi_df['sitename'].isin(aqi_sites)][remaining_features]
    test_aqi_df['datacreationdate'] = test_aqi_df['datacreationdate'].apply(lambda time: time.split(' ')[0])

    selected_columns = 'aqi,so2,co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,windspeed,winddirec,co_8hr,pm2.5_avg,pm10_avg'.split(',')

    test_aqi_df[selected_columns] = test_aqi_df[selected_columns].astype(np.float32)
    test_aqi_df = test_aqi_df.groupby(['sitename', 'datacreationdate', 'county'])[selected_columns].mean()
    test_aqi_df.reset_index(inplace = True, drop = False)

    test_aqi_df[selected_columns] = test_aqi_df[selected_columns].round(3)

    
    test_aqi_df.sort_values(by = ['sitename', 'datacreationdate'], inplace = True)
    test_aqi_df.reset_index(inplace = True, drop = True)

    for i in tqdm(range(len(test_aqi_df) - 1)):
        if test_aqi_df.loc[i, 'sitename'] != test_aqi_df.loc[i + 1, 'sitename']:
            continue
        test_aqi_df.loc[i, 'next_aqi'] = test_aqi_df.loc[i + 1, 'aqi']

    test_aqi_df.dropna(subset = ['next_aqi'], axis = 0, inplace = True)

    rainfall_dfs = dict()
    
    for site in rainfall_sites:
        df = pd.read_csv(os.path.join(RAW_FOLDER, 'rainfall', f'{site}_Precipitation.csv'))
        df['Serial'] = df['Serial'].apply(lambda x: f'{int(str(x)[:4]):04d}-{int(str(x)[4:6]):02d}-{int(str(x)[6:]):02d}')
        df = df.set_index(keys = ['Serial'])
        rainfall_dfs[site] = df

    for i in tqdm(test_aqi_df.index):
        site, date = test_aqi_df.loc[i, 'sitename'], test_aqi_df.loc[i, 'datacreationdate']
        test_aqi_df.loc[i, 'Precipitation(mm)'] = rainfall_dfs[aqi_sites[site]].loc[date, 'Precipitation(mm)']

    test_aqi_df.rename(columns={"aqi": "aqi_2"}, inplace=True)
    test_aqi_df = test_aqi_df['datacreationdate,sitename,county,aqi_2,so2,co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,windspeed,winddirec,co_8hr,pm2.5_avg,pm10_avg,Precipitation(mm),next_aqi'.split(',')]
    test_aqi_df.to_csv(os.path.join(PROCESSED_FOLDER, 'test_data.csv'), index = False)