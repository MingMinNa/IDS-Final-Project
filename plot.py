import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from utils import const


def get_correlation_coefficient(input_feature, output_feature, save_path):


    data = pd.read_csv(os.path.join(const.PROCESSED_FOLDER, 'final_result.csv'))
    
    if input_feature not in data.columns or output_feature not in data.columns:
        print(f"{input_feature} feature not in the data")
        return None
    
    data[input_feature] = pd.to_numeric(data[input_feature], errors='coerce')
    data = data[[input_feature, output_feature]].dropna().astype(np.float32)

    correlation = data[input_feature].corr(data[output_feature])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(data[input_feature], data[output_feature], alpha=0.7)
    ax.set_xlabel(input_feature)
    ax.set_ylabel(output_feature)
    ax.set_title(f"{input_feature} vs {output_feature} Correlation: {correlation:.2f}")

    ax.grid(True)
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')

    return correlation


if __name__ == '__main__':
    for input_feature in 'aqi,aqi_2,so2,co,o3,o3_8hr,pm10,pm2.5,no2,nox,no,windspeed,winddirec,co_8hr,pm2.5_avg,pm10_avg,Precipitation(mm)'.split(','):
        get_correlation_coefficient(input_feature, 'next_aqi', os.path.join(const.PROJECT_FOLDER, 'img', 'next_aqi', f'{input_feature}.png'))
        if input_feature != 'aqi_2':
            get_correlation_coefficient(input_feature, 'aqi_2', os.path.join(const.PROJECT_FOLDER, 'img', 'aqi_2', f'{input_feature}.png'))
