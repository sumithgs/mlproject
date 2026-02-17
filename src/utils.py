import os
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.custom_exception import CustomException
from src.logger import get_logger

logger = get_logger(__name__)

def load_object(file_path):
    try:
        logger.info(f"Reading pickle file from {file_path}")
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        logger.error("Error while reading pickle file!")
        raise CustomException("Error while reading pickle file",e)

def save_object(file_path,obj):
    try:
        logger.info(f"Saving the pickle file {file_path}")
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        logger.error("Error while saving pickle file!")
        raise CustomException("Error while saving pickle file",e)

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        logger.info(f"Evaluating multiple models from the list: {models.keys()}")
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)
            
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    
    except Exception as e:
        logger.error("Error while evaluating the models!")
        raise CustomException("Error while evaluating the models!",e)

