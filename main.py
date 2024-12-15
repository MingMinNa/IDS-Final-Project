import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import dataset, const
from utils.models import regression_nn, xgboost

def train_models():
    for site in tqdm(const.sitenames):

        X, y = dataset.load_dataset(sitename = site)
        X.drop(columns = ["sitename", "Year", "Day"], inplace = True)
        
        regression_path = os.path.join(const.MODEL_FOLDER, f"regression({site}).pth")
        xgboost_path = os.path.join(const.MODEL_FOLDER, f'xgboost({site}).json')
        ensemble_path = os.path.join(const.MODEL_FOLDER, f"ensemble({site}).pth")
        
        if os.path.exists(regression_path):
            regression_model = regression_nn.load_model(X.shape[1], 1, regression_path)
        else:
            regression_model, _, _ = regression_nn.build_model(X, y, regression_path, epochs = 150)
            
        if os.path.exists(xgboost_path):
            xgboost_model = xgboost.load_model(xgboost_path)
        else:
            xgboost_model = xgboost.build_model(X, y, xgboost_path)
            
        regression_y = regression_nn.predict(regression_model, X)
        xgboost_y = xgboost.predict(xgboost_model, X)
        
        X['regression'] = regression_y.cpu()
        X['xgboost'] = xgboost_y
        
        ensemble_regression, train_losses, val_losses = regression_nn.build_model(X, y, ensemble_path, epochs = 150)
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].plot(range(len(train_losses)), train_losses, label='Train Loss', color='blue')
        ax[0].set_title('Train Loss')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        ax[0].set_ylim(train_losses[-1] / 2, train_losses[-1] * 2)
        ax[0].grid(True)

        ax[1].plot(range(len(val_losses)), val_losses, label='Validation Loss', color='orange')
        ax[1].set_title('Validation Loss')
        ax[1].set_xlabel('Epochs')
        ax[1].set_ylabel('Loss')
        ax[1].legend()
        ax[1].set_ylim(val_losses[-1] / 2, val_losses[-1] * 2)
        ax[1].grid(True)

        plt.tight_layout()

        plt.savefig(os.path.join(const.MODEL_FOLDER, 'img', 'train', f"ensemble_losses({site}).png"))


def test_models(only_rain = False):

    for site in tqdm(const.sitenames):

        X, y = dataset.load_dataset(sitename = site, is_train = False)
        X.drop(columns = ["sitename", "Year", "Day"], inplace = True)

        if only_rain:
            filter_list = (X['Precipitation(mm)'] > 0.0)
            X, y = X[filter_list], y[filter_list]
            
        regression_path = os.path.join(const.MODEL_FOLDER, f"regression({site}).pth")
        xgboost_path = os.path.join(const.MODEL_FOLDER, f'xgboost({site}).json')
        ensemble_path = os.path.join(const.MODEL_FOLDER, f"ensemble({site}).pth")
        
        regression_model = regression_nn.load_model(X.shape[1], 1, regression_path)
        xgboost_model = xgboost.load_model(xgboost_path)
        ensemble_model = regression_nn.load_model(X.shape[1] + 2, 1 , ensemble_path)

        regression_y = regression_nn.predict(regression_model, X)
        xgboost_y = xgboost.predict(xgboost_model, X)
        X['regression'] = regression_y.cpu()
        X['xgboost'] = xgboost_y

        ensemble_y = regression_nn.predict(ensemble_model, X)
        error = np.abs(ensemble_y.cpu().squeeze().numpy() - y.to_numpy())
        mae = np.sum(error) / len(y)

        plt.figure(figsize=(8, 6))
        plt.hist(error, bins = 10, color = "skyblue", edgecolor = "black", alpha=0.7)
        plt.title(f"Error Distribution (MAE = {mae:.3f})", fontsize = 16)
        plt.xlabel("Error", fontsize = 14)
        plt.ylabel("Frequency", fontsize = 14)
        plt.grid(axis = "y", linestyle = "--", alpha = 0.7)

        img_path = os.path.join(const.MODEL_FOLDER, 'img', 'test', 'only_rain' if only_rain else 'all', f"ensemble_losses({site}).png")
        plt.savefig(img_path)

if __name__ == '__main__':
    # train_models()
    test_models(only_rain = True)
