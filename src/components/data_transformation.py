# used for transforming or standarization of data like categorical data or numerical data
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer # used for creating pipeline
from sklearn.impute import SimpleImputer # for missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config.paths_config import *
from src.utils import save_object
from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

"""
We use @dataclass decorator , because inside any traditional class, to define the class variables you basically use _init_ ,  but if we use this @dataclass decorator,  it enables us to define the class variable directly.
"""

class DataTransformation:
    def __init__(self):
        os.makedirs(PROCESSED_DIR,exist_ok=True)
        self.preprocessor_ob_file_path = PROCESSED_OBJ_FILE_PATH
    
    def get_data_transformer_object(self):
        """
        This function is for data transformation like encoding categorical values, normalising numerical values, etc.
        """
        try:
            numerical_columns = ["writing_score",'reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            # normalising numerical values
            # handling missing values imputer
            # scaling values to make them in range 0 to 1
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            # normalising categorical values
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    # ("scaler",StandardScaler()) # if we want to use this then use it with_mean = false
                    # not required because categorical values are 0 and 1 if we use standardisation there is a chance it will convert or compress some 0's and 1's in other values like 0.9 or 0.8 etc.
                ]
            )
            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            logger.info("Numerical columns standard scaling completed!")
            logger.info("Categorical columns encoding completed!")
            return preprocessor
        
        except Exception as e:
            logger.error("Error while transforming the columns!")
            raise CustomException("Error while transforming the columns!",e)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Reading the training and testing data completed!")

            logger.info("Obtaining preprocessing object file")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ["writing_score",'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )


            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            feature_names = preprocessing_obj.get_feature_names_out()
            columns = list(feature_names) + [target_column_name]

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logger.info(f"Saving preprocessing object file {self.preprocessor_ob_file_path}")
            # saving the pickle file
            save_object(
                file_path=self.preprocessor_ob_file_path,
                obj=preprocessing_obj
            )
            logger.info(f"Saving preprocessed data to {PROCESSED_TRAIN_DATA_PATH} and {PROCESSED_TEST_DATA_PATH}!")
            train_df_processed = pd.DataFrame(train_arr, columns=columns)
            test_df_processed  = pd.DataFrame(test_arr, columns=columns)
            train_df_processed.to_csv(PROCESSED_TRAIN_DATA_PATH, index=False)
            test_df_processed.to_csv(PROCESSED_TEST_DATA_PATH, index=False)


        except Exception as e:
            logger.error("Failed while during data preprocessing stage!")
            raise CustomException("Failed while during data preprocessing stage!",e)
    
    def process(self):
        try:
            logger.info("Initiating Data Preprocessing stage!")
            self.initiate_data_transformation(train_path=TRAIN_FILE_PATH,test_path=TEST_FILE_PATH)

        except Exception as e:
            logger.error("Error during data preprocessing stage!")
            raise CustomException("Error during data preprocessing stage!",e)

if __name__=="__main__":
    obj = DataTransformation()
    obj.process()
