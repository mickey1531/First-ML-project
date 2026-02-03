from source.MLProject.logger import logging
from source.MLProject.exception import CustomException
from source.MLProject.components.data_ingestion import DataIngestion
from source.MLProject.components.data_transformation import DataTransformation
from source.MLProject.components.model_trainer import ModelTraining
import sys

# train_arr, test_arr, self.data_transformation_config.preprocessor_obj_filepath

if __name__ == "__main__":
    logging.info("The exeuction has started")


    try:

        data_ingeston = DataIngestion()
        train_path , test_path = data_ingeston.initiate_data_ingestion()


        data_transformation = DataTransformation()
        train_arr, test_arr ,_ =  data_transformation.initiate_data_transformation(train_path, test_path)

        data_modeling = ModelTraining()
        score = data_modeling.initiate_model_trainer(train_arr,test_arr)
        print(score)



      
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys) # type: ignore
