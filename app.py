from source.MLProject.logger import logging
from source.MLProject.exception import CustomException
from source.MLProject.components.data_ingestion import DataIngestion
import sys



if __name__ == "__main__":
    logging.info("The exeuction has started")


    try:

        data_ingeston = DataIngestion()
        data_ingeston.initiate_data_ingestion()
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys) # type: ignore
