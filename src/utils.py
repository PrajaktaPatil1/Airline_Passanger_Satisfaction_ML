import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import roc_auc_score,classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            # print(model)

            # para=param[list(models.keys())[i]]
            # print(para)

            # gs = GridSearchCV(model,para,cv=3)
            # gs.fit(X_train,y_train)

            # model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

    

            # y_train_pred = model.predict(X_train)

            # y_test_pred = model.predict(X_test)

            # train_model_score = classification_report(y_train, y_train_pred)

            # test_model_score = classification_report(y_test, y_test_pred)

            # report[list(models.keys())[i]] = test_model_score
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = roc_auc_score(y_train, y_train_pred)
            test_model_score = roc_auc_score(y_test, y_test_pred)

            train_classification_report = classification_report(y_train, y_train_pred)
            test_classification_report = classification_report(y_test, y_test_pred)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = {
                'roc_auc_score': test_model_score,
                'classification_report': test_classification_report,
                'accuracy': test_accuracy
            }

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)