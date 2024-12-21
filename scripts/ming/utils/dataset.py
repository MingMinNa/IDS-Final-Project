import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

try:    import const
except: from . import const

def preprocess(X):

    scaler = StandardScaler()
    for feature in X.columns:
        if feature in ['sitename', 'Year', 'Month', 'Day']: continue
        X.loc[:, [feature]] = scaler.fit_transform(X.loc[:, [feature]])

    return X

def load_dataset(sitename = None, is_train = True):

    if (sitename is not None) and (sitename not in const.sitenames):
        raise ValueError(f'The sitename({sitename}) is not in the dataset')
    
    if is_train:
        data = pd.read_csv(os.path.join(const.PROCESSED_FOLDER, 'train_data.csv'))
    else:
        data = pd.read_csv(os.path.join(const.PROCESSED_FOLDER, 'test_data.csv'))

    '''
    All features:
        datacreationdate,sitename,county,aqi,aqi_2,so2,
        co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,windspeed,winddirec,
        co_8hr,pm2.5_avg,pm10_avg,Precipitation(mm),next_aqi
    '''


    # '2017-04-03' -> Year: 2017, Month: 4, Day: 3
    for idx, new_feature in enumerate(['Year', 'Month', 'Day']):
        data[new_feature] = data['datacreationdate'].apply(lambda date:int(date.split('-')[idx]))

    # features to remove
    remove_features = 'windspeed,winddirec,county'.split(',') + (['aqi'] if is_train else [])
    data.drop(columns = remove_features + ['datacreationdate'], inplace = True)

    data = data[data[['next_aqi']].notna().any(axis=1)]
    data = data.dropna(subset = 'so2,co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,co_8hr,pm2.5_avg,pm10_avg'.split(','), how = 'all').reset_index(drop = True)

    
    # if the feature value is less than 0, set the value to 0
    for col in data.columns:
        if col == 'sitename': continue
        data[col] = data[col].astype(np.float32)
        data[col] = data[col].clip(lower = 0)


    # Sort features to have the same feature order every program execution
    remaining_features = sorted([feature for feature in data.columns if feature != 'next_aqi'])

    # no site specified, use the data of all sites
    if sitename is None:

        # if the feature value is nan, use the average value of the same site on the same Year and Month
        for site in tqdm(const.sitenames, desc = 'Handle nan values'):
            na_data = data[data.isna().any(axis=1)]
            na_data = na_data[na_data['sitename'] == site]

            for i in na_data.index:
                cond_1 = (data['sitename'] == site)
                cond_2 = (data['Year'] == na_data.loc[i, 'Year'])
                cond_3 = (data['Month'] == na_data.loc[i, 'Month'])
                cond_data = data[cond_1 & (cond_2) & (cond_3)]

                for col in data.columns:
                    if col == 'sitename' or not pd.isna(na_data.loc[i, col]): continue
                    data.loc[i, col] = cond_data[col].mean()
        
        data.reset_index(drop = True, inplace = True)
        return data[remaining_features], data['next_aqi']
    else:

        na_data = data[data.isna().any(axis=1)]
        na_data = na_data[na_data['sitename'] == sitename]

        for i in na_data.index:
            cond_1 = (data['sitename'] == sitename)
            cond_2 = (data['Year'] == na_data.loc[i, 'Year'])
            cond_3 = (data['Month'] == na_data.loc[i, 'Month'])
            cond_data = data[cond_1 & (cond_2) & (cond_3)]

            for col in data.columns:
                if col == 'sitename' or not pd.isna(na_data.loc[i, col]): continue
                data.loc[i, col] = cond_data[col].mean()
        
        data = data[data['sitename'] == sitename]
        data.reset_index(drop = True, inplace = True)
        return data[remaining_features], data['next_aqi']

if __name__ == '__main__':
    pass