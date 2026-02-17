import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass # used for creating class variables

from src.custom_exception import CustomException
from src.logger import get_logger
from config.paths_config import *

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self):
        self.train_data_path = TRAIN_FILE_PATH
        self.test_data_path = TEST_FILE_PATH
        self.raw_data_path = RAW_FILE_PATH
    
    def initiate_data_ingestion(self):
        logger.info("Initiating Data Ingestion!")

        try:
            logger.info(f"Reading the dataset from {self.raw_data_path}")
            df = pd.read_csv(self.raw_data_path)

            logger.info("Train-test split initiated")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.train_data_path,index=False,header=True)

            test_set.to_csv(self.test_data_path,index=False,header=True)

            logger.info("Data Ingestion state Completed!")
            # return(
            #     self.ingestion_config.train_data_path,
            #     self.ingestion_config.test_data_path
            # )
        except Exception as e:
            logger.error("Error while splitting data")
            raise CustomException("Failed to split data into train and test set",e)

if __name__=="__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()