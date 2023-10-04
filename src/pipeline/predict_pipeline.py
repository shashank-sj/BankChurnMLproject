import os
import sys 
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,geography,gender,age,tenure,balance,num_of_products,has_cr_card,is_active_member,estimated_salary,credit_score) :
        self.geography = geography
        self.gender = gender
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.num_of_products = num_of_products
        self.has_cr_card = has_cr_card
        self.is_active_member = is_active_member
        self.estimated_salary = estimated_salary
        self.credit_score = credit_score

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "CreditScore": [self.credit_score],
                "Geography":[self.geography],
                "Gender": [self.gender],
                "Age": [self.age],
                "Tenure": [self.tenure],
                "Balance": [self.balance],
                "NumOfProducts": [self.num_of_products],
                "HasCrCard": [self.has_cr_card],
                "IsActiveMember": [self.is_active_member],
                "EstimatedSalary": [self.estimated_salary],
                
            }    
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)
   