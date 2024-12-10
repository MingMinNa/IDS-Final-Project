import os
import pandas as pd
from tqdm import tqdm

PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
RAW_FOLDER = os.path.join(PROJECT_FOLDER, 'data', 'raw')
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, 'data', 'processed')

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

    rainfall_dfs = dict()
    
    for site in rainfall_sites:
        df = pd.read_csv(os.path.join(RAW_FOLDER, 'rainfall', f'{site}_Precipitation.csv'))
        df['Serial'] = df['Serial'].apply(lambda x: f'{int(str(x)[:4]):04d}-{int(str(x)[4:6]):02d}-{int(str(x)[6:]):02d}')
        df = df.set_index(keys = ['Serial'])
        rainfall_dfs[site] = df

    merged_aqi_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'AQI', 'result.csv'))

    for i in tqdm(merged_aqi_df.index):
        site, date = merged_aqi_df.loc[i, 'sitename'], merged_aqi_df.loc[i, 'datacreationdate']
        merged_aqi_df.loc[i, 'Precipitation(mm)'] = rainfall_dfs[aqi_sites[site]].loc[date, 'Precipitation(mm)']

    merged_aqi_df.to_csv(os.path.join(PROCESSED_FOLDER, 'final_result.csv'), index = False)
    quit()