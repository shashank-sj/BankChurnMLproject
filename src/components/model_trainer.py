import os 
import sys
from src.utils import save_object
from src.utils import evaluate_models
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            logging.info("Spliting data into train and test")
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            ) 
            models = {
                    'Random Forest': RandomForestClassifier(),
                    'AdaBoost': AdaBoostClassifier(),
                    'Gradient Boosting': GradientBoostingClassifier(),
                    'K-Nearest Neighbors': KNeighborsClassifier(),
                    'Logistic Regression': LogisticRegression(),
                    'Decision Tree': DecisionTreeClassifier(),
                    'Multinomial Naive Bayes': MultinomialNB(),
                    }
            model_report: dict = evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            scoring_metric = "Precision"
            best_model_name = max(model_report, key=lambda model: model_report[model][scoring_metric])
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name][scoring_metric]
            if best_model_score <0.6:
                raise CustomException("No best model found")
            logging.info("Best model found")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted = best_model.predict(X_test)
            accuracy = accuracy_score(y_test,predicted)
            precision = precision_score(y_test,predicted)
            return accuracy,precision,best_model_name
        except Exception as e:
            raise CustomException(e, sys)

            