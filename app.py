from source.MLProject.logger import logging
from source.MLProject.exception import CustomException
from source.MLProject.components.data_ingestion import DataIngestion
from source.MLProject.components.data_transformation import DataTransformation
import sys



if __name__ == "__main__":
    logging.info("The exeuction has started")


    try:

        data_ingeston = DataIngestion()
        train_path , test_path = data_ingeston.initiate_data_ingestion()
        data_transformation = DataTransformation()

        data_transformation.initiate_data_transformation(train_path, test_path)





        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys) # type: ignore
