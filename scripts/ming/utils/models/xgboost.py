import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from scipy.stats import uniform, randint

def build_model(X, y, save_path = None, random_seed = 42):
    
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )
    
    param_dist = {
        'learning_rate': uniform(0.01, 0.3),  
        'max_depth': randint(3, 10),          
        'n_estimators': randint(100, 501),    
        'subsample': uniform(0.6, 0.4),       
        'colsample_bytree': uniform(0.6, 0.4),
        'reg_alpha': uniform(0, 1),           
        'reg_lambda': uniform(1, 2),          
    }
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:absoluteerror',  
        random_state=random_seed
    )
    
    random_search = RandomizedSearchCV(
        estimator = xgb_model,
        param_distributions = param_dist,
        n_iter = 50,  
        scoring = 'neg_mean_absolute_error', 
        cv = 3,  
        verbose = 1,
        random_state = random_seed,
        n_jobs = -1 
    )
    
    random_search.fit(X_train_split, y_train_split)
    print("Best Parameters:", random_search.best_params_)
    print("Best CV Score (negative MAE):", random_search.best_score_)
    
    best_model = random_search.best_estimator_
    best_model.fit(X, y)
    
    val_predictions = best_model.predict(X_val_split)
    val_mae = mean_absolute_error(y_val_split, val_predictions)
    print(f"[best xgb model] Validation MAE: {val_mae:.4f}")
    
    if save_path is not None:
        save_model(save_path, best_model)
        
    return best_model


def predict(xgboost_model, input_val):
    y_predicts = xgboost_model.predict(input_val)
    return y_predicts

def load_model(model_path):
    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(model_path)
    return xgboost_model

def save_model(model_path, xgboost_model):
    xgboost_model.save_model(model_path)
    return

if __name__ == "__main__":
    pass