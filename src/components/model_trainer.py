import os
import sys
from dataclasses import dataclass

import sklearn

### sklearn preprocessing tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score


# Error Metrics 
from sklearn.metrics import r2_score  # r2 square
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


### Machine learning classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier  # stochastic gradient descent classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lightgbm
from sklearn.ensemble import AdaBoostClassifier

# Cross-validation
from sklearn.model_selection import cross_val_score

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            models = {
                "LogisticRegression": LogisticRegression(max_iter=1000),
                "GaussianNB": GaussianNB(),
                "KNeighborsClassifier": KNeighborsClassifier(),
                "SGDClassifier": SGDClassifier(loss='modified_huber', n_jobs=-1, random_state=42),
                # "DecisionTreeClassifier": DecisionTreeClassifier(),
                # "RandomForestClassifier": RandomForestClassifier(),
                "AdaBoostClassifier": AdaBoostClassifier(),
                "GradientBoostingClassifier": GradientBoostingClassifier(),
                # "XGBClassifier": XGBClassifier(),
                "LGBMClassifier": lightgbm.LGBMClassifier(),
            }
            # params = {
            #     "LogisticRegression": {
            #         "penalty": ["l1", "l2"],
            #         "C": [0.01, 0.1, 1, 10]
            #     },
            #     "GaussianNB": {
            #                         'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            #                     },
            #     "LinearSVC" : {
            #                         'C': [0.1, 1, 10],
            #                         'penalty': ['l1', 'l2'],
            #                         'loss': ['hinge', 'squared_hinge']
            #                     },

            #     "KNeighborsClassifier" : {
            #                     'n_neighbors': [3, 5, 7],
            #                     'weights': ['uniform', 'distance'],
            #                     'algorithm': ['auto', 'ball_tree', 'kd_tree']
            #                 },
            #     "SGDClassifier" : {
            #                 'loss': ['hinge', 'log', 'modified_huber'],
            #                 'penalty': ['l2', 'l1', 'elasticnet'],
            #                 'alpha': [0.0001, 0.001, 0.01],
            #                 'max_iter': [1000, 2000, 3000]
            #             },


            #     "DecisionTreeClassifier": {
            #         "criterion": ["gini", "entropy"],
            #         "splitter": ["best", "random"],
            #         "max_depth": [8, 15, 20, 25],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 3, 5],
            #         "max_features": ["auto", "sqrt", "log2"]
            #     },
            #     "RandomForestClassifier": {
            #         "n_estimators": [100, 200, 300],
            #         "criterion": ["gini", "entropy"],
            #         "max_depth": [8, 15, 20],
            #         "min_samples_split": [2, 5, 10],
            #         "min_samples_leaf": [1, 3, 5],
            #         "max_features": ["auto", "sqrt", "log2"]
            #     },
            #     "AdaBoostClassifier": {
            #         "n_estimators": [200, 300],
            #         "learning_rate": [1, 0.1, 0.01, 0.001]
            #     },
            #     "GradientBoostingClassifier": {
            #         "n_estimators": [400, 500],
            #         "learning_rate": [0.1],
            #         "max_depth": [3, 5, 8]
            #     },
            #     "KNeighborsClassifier": {
            #         "n_neighbors": [5, 10, 15],
            #         "weights": ["uniform", "distance"],
            #         "algorithm": ["auto", "ball_tree", "kd_tree"]
            #     },
            #     "LGBMClassifier": {
            #         "n_estimators": [100, 200, 300],
            #         "learning_rate": [0.1, 0.01],
            #         "max_depth": [5, 8, 10],
            #         "num_leaves": [30, 50, 100],
            #         "boosting_type": ["gbdt", "dart"]
            #     }
            # }


            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                                     models=models)

            best_model_score = 0.0
            best_model_name = None
            best_metric = None

            for model_name, metrics in model_report.items():
                model_score = metrics['roc_auc_score']
                if model_score > best_model_score:
                    best_model_score = model_score
                    best_model_name = model_name
                    best_metric = metrics

            best_model = models[best_model_name]
            print(f"Best Model Found, Model Name is:---------> {best_model_name}")
            logging.info(f"Best Model Found, Model Name is: {best_model_name}, Roc AUC: {best_model_score}, {best_metric}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            roc_auc_score_ = roc_auc_score(y_test, predicted)
            classification_report_ = classification_report(y_test, predicted)
            accuracy_score_ = accuracy_score(y_test, predicted)

            return "accuracy_score_-------->", accuracy_score_, "roc_auc_score_-------->:", roc_auc_score_

 
        except Exception as e:
            raise CustomException(e, sys)