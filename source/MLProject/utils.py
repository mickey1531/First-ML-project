import os
import sys
from source.MLProject.logger import logging
from source.MLProject.exception import CustomException
import pandas as pd
from dotenv import load_dotenv
import pymysql



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