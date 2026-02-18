# train diffferent models and see the accuracy

import os
import sys
import pandas as pd

# from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.custom_exception import CustomException
from src.logger import get_logger
from src.utils import save_object,evaluate_models
from config.paths_config import *

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self,train_path, test_path):
        self.train_path = train_path
        self.test_path  = test_path

    def load_and_split(self):
        """Load CSVs, convert to NumPy arrays, and split features/target"""
        train_df = pd.read_csv(self.train_path)
        test_df  = pd.read_csv(self.test_path)

        train_array = train_df.to_numpy()
        test_array  = test_df.to_numpy()

        return train_array,test_array
    
    def initiate_model_trainer(self):
        try:
            logger.info("Splitting training and test input data")
            train_array,test_array = self.load_and_split()
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                # "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                # "CatBoosting Regressor":{
                #     'depth': [6,8,10],
                #     'learning_rate': [0.01, 0.05, 0.1],
                #     'iterations': [30, 50, 100]
                # },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            model_report:dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models = models,params=params)
            
            # get best model score
            best_model_score = max(sorted(model_report.values()))

            logger.info(f"Evaluation report of models: {model_report}")

            #get best model name
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            # if best_model_score<0.6:
            #     raise CustomException("No best model found",sys)
            logger.info(f"Best found model on both training and testing dataset {best_model_name}")

            save_object(
                file_path=MODEL_OUTPUT_PATH,
                obj=best_model
            )
            logger.info(f"Saved the model to {MODEL_OUTPUT_PATH}")
        
        except Exception as e:
            logger.error("Error while Model saving stage!")
            raise CustomException("Error while Model saving stage!",e)

if __name__=="__main__":
    obj = ModelTrainer(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH)
    obj.initiate_model_trainer()

