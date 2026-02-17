from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from config.paths_config import *

if __name__=="__main__":
    ## Data ingestion
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_ingestion()

    ## Data Transformation
    data_transformation = DataTransformation()
    data_transformation.process()

    ## Model Training
    model_training = ModelTrainer(PROCESSED_TRAIN_DATA_PATH,PROCESSED_TEST_DATA_PATH)
    model_training.initiate_model_trainer()