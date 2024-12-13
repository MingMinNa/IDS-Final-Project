import pandas as pd

file_path = 'data/processed/final_result.csv'
data = pd.read_csv(file_path)

data = data[data['county'] == '高雄市']

data['month'] = pd.to_datetime(data['datacreationdate']).dt.month
data['day'] = pd.to_datetime(data['datacreationdate']).dt.day
data['year'] = pd.to_datetime(data['datacreationdate']).dt.year

data = data.dropna(subset=['next_aqi'])

data = data.drop(columns=['datacreationdate'])
data = data.drop(columns=['county'])

data_2024 = data[data['year'] == 2024]
data_other_years = data[data['year'] != 2024]

data_2024 = data_2024.drop(columns=['year'])
data_other_years = data_other_years.drop(columns=['year'])

data_2024.to_csv('data/processed/data_2024.csv', index=False)
data_other_years.to_csv('data/processed/data_other_years.csv', index=False)