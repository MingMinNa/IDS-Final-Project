import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import re


PROJECT_FOLDER = os.path.dirname(os.path.dirname(__file__))
RAW_FOLDER = os.path.join(PROJECT_FOLDER, 'data', 'raw')
PROCESSED_FOLDER = os.path.join(PROJECT_FOLDER, 'data', 'processed')
TMP_FOLDER = os.path.join(PROCESSED_FOLDER, 'tmp')

def remove_folder(folder_path):
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        os.rmdir(folder_path)
    return

def history_AQI_filter(remaining_features: list[str]):
    
    aqi_df = pd.read_csv(os.path.join(RAW_FOLDER, 'AQI.csv'))
    stations = set(aqi_df['Location'])
    del aqi_df

    os.makedirs(TMP_FOLDER, exist_ok = True)

    for filename in tqdm(os.listdir(os.path.join(RAW_FOLDER, 'AQI'))):
        output_path = os.path.join(PROCESSED_FOLDER, 'tmp', filename)
        if 'csv' not in filename:            continue
        elif os.path.exists(output_path):    
            print(f"路徑 {output_path} 已經存在")
            continue

        history_aqi_df = pd.read_csv(os.path.join(RAW_FOLDER, 'AQI', filename))

        # remain the target stations && keep remaining features(remove useless features)
        filtered_df = history_aqi_df[history_aqi_df['sitename'].isin(stations)][remaining_features]

        # extract date from 'datacreationdate' feature, for example, convert '2024-01-01 00:00' to '2024-01-01'
        filtered_df['datacreationdate'] = filtered_df['datacreationdate'].apply(lambda time: time.split(' ')[0])
        filtered_df.to_csv(output_path, index = False)

def generate_result(remaining_features: list[str]):
    
    aqi_df = pd.read_csv(os.path.join(RAW_FOLDER, 'AQI.csv'))
    result_path = os.path.join(PROCESSED_FOLDER, 'result.csv')
    # use remaining features as columns & reorder the columns
    if not os.path.exists(result_path):
        result_df = pd.DataFrame(columns = remaining_features + ['aqi_2'])['datacreationdate,sitename,county,aqi,aqi_2,so2,co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,windspeed,winddirec,co_8hr,pm2.5_avg,pm10_avg'.split(',')]
    else:
        result_df = pd.read_csv(result_path)
    
    tmp_dfs = dict()
    for filename in os.listdir(TMP_FOLDER):
        month = re.search(r'\((\d{4}-\d{2})\)', filename).group(1)
        tmp_dfs[month] = pd.read_csv(os.path.join(TMP_FOLDER, filename))
    
    for i in tqdm(range(len(result_df), len(aqi_df.index))):
        month = '-'.join(aqi_df.loc[i, 'Date'].split('-')[:-1])
        tmp_df = tmp_dfs[month]
        tmp_df = tmp_df[(tmp_df['sitename'] == aqi_df.loc[i, 'Location']) & (tmp_df['datacreationdate'] == aqi_df.loc[i, 'Date'])].reset_index(drop = True)
        
        for feature in tmp_df.columns:
            try:
                tmp_df[feature] = tmp_df[feature].astype(np.float32)
                tmp_df[feature] = tmp_df[feature].fillna(tmp_df[feature].mean())
                result_df.loc[i, feature] = tmp_df[feature].mean()
            except: # 'sitename' or 'country' features 
                result_df.loc[i, feature] = tmp_df.loc[0, feature]
        result_df.loc[i, 'aqi_2'] = aqi_df.loc[i, 'AQI']
        
        if i % 1000 == 0:
            result_df.to_csv(result_path, index = False)
    
    result_df.to_csv(result_path, index = False)
    print('result.csv was created successfully.')


    # other case: complement information of missing date(2021-05-01 ~ 2021-05-31)
    sites = set(aqi_df['Location'])
    missing_df = pd.read_csv(os.path.join(RAW_FOLDER, 'missing.csv'))
    missing_df['Date'] = pd.to_datetime(missing_df['Date']).dt.date
    idx = len(result_df)
    for i in missing_df.index:
        tmp_date = missing_df.loc[i, 'Date']
        # site 
        for site in sites:
            month = f"{tmp_date.year:04d}-{tmp_date.month:02d}"
            tmp_df = tmp_dfs[month]
            tmp_df = tmp_df[(tmp_df['sitename'] == site) & (tmp_df['datacreationdate'] == f"{month}-{tmp_date.day:02d}")].reset_index(drop = True)
            
            for feature in tmp_df.columns:
                try:
                    tmp_df[feature] = tmp_df[feature].astype(np.float32)
                    tmp_df[feature] = tmp_df[feature].fillna(tmp_df[feature].mean())
                    result_df.loc[idx, feature] = tmp_df[feature].mean()
                except: # 'sitename' or 'country' features 
                    result_df.loc[idx, feature] = tmp_df.loc[0, feature]
            result_df.loc[idx, 'aqi_2'] = result_df.loc[idx, 'aqi'] 
            idx += 1
        result_df.to_csv(result_path, index = False)

    print('Information supplement from 2021-05-01 to 2021-05-31 completed')
    return

def result_handle():
    result_path = os.path.join(PROCESSED_FOLDER, 'result.csv')
    result_df = pd.read_csv(result_path)
    aqi_df = pd.read_csv(os.path.join(RAW_FOLDER, 'AQI.csv'))

    for i in aqi_df.index:
        if pd.isna(result_df.loc[i, 'datacreationdate']):
            result_df.loc[i, 'datacreationdate'] = aqi_df.loc[i, 'Date']
            result_df.loc[i, 'sitename'] = aqi_df.loc[i, 'Location']

    # round the values
    for feature in result_df.columns:
        try:    result_df[feature] = result_df[feature].round(3)
        except: pass

    result_df.sort_values(by = ['datacreationdate', 'sitename'], ascending = True, inplace = True)
    result_df.to_csv(result_path, index = False)
    return

def next_aqi():
    final_result_df = pd.read_csv(os.path.join(PROCESSED_FOLDER, 'final_result.csv'))

    final_result_df.sort_values(by = ['sitename', 'datacreationdate'], inplace = True)
    final_result_df.reset_index(inplace = True, drop = True)

    for i in tqdm(range(len(final_result_df) - 1)):
        if final_result_df.loc[i, 'sitename'] != final_result_df.loc[i + 1, 'sitename']:
            continue
        final_result_df.loc[i, 'next_aqi'] = final_result_df.loc[i + 1, 'aqi_2']
    
    final_result_df.to_csv(os.path.join(PROCESSED_FOLDER, 'test.csv'), index = False)

if __name__ == '__main__':

    raw_AQI_features = 'sitename,county,aqi,pollutant,status,so2,co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,windspeed,winddirec,datacreationdate,unit,co_8hr,pm2.5_avg,pm10_avg,so2_avg,longitude,latitude,siteid'.split(',')

    # most of 'so2_avg' is NaN, don't treat this feature as an input feature
    removed_features = 'status,longitude,latitude,siteid,pollutant,unit,so2_avg'.split(',')
    remaining_features = [feature for feature in raw_AQI_features if feature not in removed_features]

    # history_AQI_filter(remaining_features)
    # generate_result(remaining_features)
    # result_handle()

    next_aqi()

