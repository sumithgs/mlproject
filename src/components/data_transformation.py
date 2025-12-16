# used for transforming or standarization of data like categorical data or numerical data
import sys
import os
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer # used for creating pipeline
from sklearn.impute import SimpleImputer # for missing values
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import save_object

"""
We use @dataclass decorator , because inside any traditional class, to define the class variables you basically use _init_ ,  but if we use this @dataclass decorator,  it enables us to define the class variable directly.
"""
@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
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
            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read tarin and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_columns = ["writing_score",'reading_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")
            # saving the pickle file
            save_object(

                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomException(e,sys)