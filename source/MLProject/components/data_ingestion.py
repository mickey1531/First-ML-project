import os
import sys
from source.MLProject.logger import logging
from source.MLProject.exception import CustomException
import pandas as pd
import numpy as np
from source.MLProject.utils import read_sql_data
from sklearn.model_selection import train_test_split

class DataIngestionConfig:
    train_data_path = os.path.join("artifacts","train.csv")
    test_data_path = os.path.join("artifacts","test.csv")
    raw_data_path = os.path.join("artifacts","raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            # Reading the data from My sql
            logging.info("Reading completed mysql database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # data = read_sql_data()
            data = pd.read_csv(os.path.join("notebook/data","raw.csv"))     # This is added to test the data transformation and to not pull data from sql again
            data.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set,test_set = train_test_split(data, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion is completed.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )




        except Exception as e:
            raise CustomException(e,sys) # type: ignore


