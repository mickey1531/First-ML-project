import os
import sys
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error,r2_score, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from xgboost import XGBRegressor

from source.MLProject.utils import evalutate_model, save_object
from source.MLProject.logger import logging
from source.MLProject.exception import CustomException

import mlflow
import mlflow.sklearn
from urllib.parse import urlparse


class ModelTrainerConfig:
    trained_model_filepath = os.path.join("artifacts","model.pkl")

class ModelTraining:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)
        return rmse, mae, r2



    def initiate_model_trainer(self,train_arr, test_arr):
        try:
            logging.info("Split training and test input data")

            x_train = train_arr[:,:-1]
            y_train = train_arr[:,-1]
            x_test = test_arr[:,:-1]
            y_test = test_arr[:,-1]

            models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree" : DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(),
            "AdaBoost Regressor": AdaBoostRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
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
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            logging.info("Evalutating the model")
            model_result = evalutate_model(x_train,y_train,x_test,y_test,models, params)

            best_model_score = max(model_result.values())
            best_model_name = list(model_result.keys())[list(model_result.values()).index(best_model_score)]

            best_model = models[best_model_name]



            # print(f"This is the best model : {best_model_name}")
            
            # model_names =  list(models.keys())

            # actual_model = ""
            # for model_i in model_names:
            #     if best_model_name == model_i:
            #         actual_model = actual_model + model_i

            # best_params = params[actual_model]
            
            # # mlflow.set_registry_uri("https://dagshub.com/mickey1531/First-ML-project.mlflow")
            # trackng_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            # with mlflow.start_run():
            #     predicted_qualities = best_model.predict(x_test)

            #     rmse,mae,r2 = self.eval_metrics(y_test,predicted_qualities)

            #     mlflow.log_params(best_params)
            #     mlflow.log_metric("rmse", rmse)
            #     mlflow.log_metric("mae", mae)
            #     mlflow.log_metric("r2", r2)
                
            #     if trackng_url_type_store != "file":
            #         mlflow.sklearn.log_model(best_model,"model",registered_model_name=actual_model)
            #     else:
            #         mlflow.sklearn.log_model(best_model,"model")

            
            
            
            
            if best_model_score<0.6:
                raise CustomException("No best model found",sys)  # type: ignore
            logging.info(f"Best model found on both training and test dataset")

            save_object(self.model_trainer_config.trained_model_filepath, best_model)

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square



        except Exception as e:
            raise CustomException(e,sys) # type: ignore
