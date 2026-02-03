import sys
import os
from source.MLProject.logger import logging
from source.MLProject.exception import CustomException

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from source.MLProject.utils import save_object

class Datatransformationconfig:
    preprocessor_obj_filepath = os.path.join("artifacts","preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = Datatransformationconfig


    def get_transformer_object(self):
        try:
            num_feature = ["writing_score","reading_score"]
            cat_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot",OneHotEncoder())
            ])

            logging.info(f"Categorical Column : {cat_features}")
            logging.info(f"Numerical Column : {num_feature}")

            preprocessor = ColumnTransformer([
                ("numerical", num_pipeline,num_feature),
                ("categorical", cat_pipeline,cat_features) 
            ])


            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)  #type: ignore


    def initiate_data_transformation(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading the train and test file")

            preprocessing_obj = self.get_transformer_object()

            target_col_name = ["math_score"]

            input_feature_train = train_df.drop(target_col_name, axis = 1)
            target_feature_train = train_df[target_col_name]

            input_feature_test = test_df.drop(target_col_name, axis = 1)
            target_feature_test = test_df[target_col_name]

            logging.info("Applying preprocessing on training and test dataframe")

            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test)

            train_arr = np.c_[input_feature_train_array, np.array(target_feature_train)]
            test_arr = np.c_[input_feature_test_array, np.array(target_feature_test)]

            logging.info(f"Saved preprocessing object")

            save_object(file_path= self.data_transformation_config.preprocessor_obj_filepath, obj=preprocessing_obj)


            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_filepath


        except Exception as e:
            raise CustomException(e,sys) #type: ignore
