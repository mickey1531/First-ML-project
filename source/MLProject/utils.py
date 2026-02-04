import os
import sys
from source.MLProject.logger import logging
from source.MLProject.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import pickle
import numpy as np


load_dotenv()

myhost = os.getenv("host")
myuser = os.getenv("user")
pw = os.getenv("password")
db = os.getenv("db")


# print(myhost,myuser,pw,db)

def read_sql_data():
    logging.info("Reading SQL database started")
    try:
        mydb = pymysql.connect(
            host=myhost,
            user=myuser,
            database= db,
            password= str(pw)
        )
        logging.info("Connection established with ",mydb)
        df= pd.read_sql_query("Select * from student", mydb)
        print(df.head())

        return df
    
    except Exception as ex:
        raise CustomException(ex,sys) # type: ignore
    





def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys) # type: ignore
    





def evalutate_model(x_train,y_train,x_test, y_test,models:dict,param:dict):

    try:

        report = {}
        for i in list(models.keys()):
            model = models[i]
            para = param[i]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(x_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[i] = test_model_score
        
        return report    

    except Exception as e:
        raise CustomException(e,sys) # type: ignore

   
     


