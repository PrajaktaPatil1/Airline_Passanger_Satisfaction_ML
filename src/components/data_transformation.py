import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns =['age','flight_distance', 'inflight_wifi_service',
                                'departure_arrival_time_convenient', 
                                'ease_of_online_booking','gate_location',
                                'food_and_drink','online_boarding', 
                                'seat_comfort','inflight_entertainment', 
                                'on_board_service', 'leg_room_service',
                                'baggage_handling', 'checkin_service', 
                                'inflight_service','cleanliness',
                                'departure_delay_in_minutes',
                                'arrival_delay_in_minutes',
                                    ]
            categorical_columns =['gender', 'customer_type', 'type_of_travel', 'Class']

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
               
                ("one_hot_encoder",OneHotEncoder()),
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):

            try:
                train_df=pd.read_csv(train_path)
                test_df=pd.read_csv(test_path)

                logging.info("Read train and test data completed")

                logging.info("Obtaining preprocessing object")

                preprocessing_obj=self.get_data_transformer_object()
                
                target_column_name='satisfaction'
                numerical_columns =['age','flight_distance', 'inflight_wifi_service',
                                    'departure_arrival_time_convenient', 
                                    'ease_of_online_booking','gate_location',
                                     'food_and_drink','online_boarding', 
                                     'seat_comfort','inflight_entertainment', 
                                     'on_board_service', 'leg_room_service',
                                    'baggage_handling', 'checkin_service', 
                                    'inflight_service','cleanliness',
                                    'departure_delay_in_minutes',
                                    'arrival_delay_in_minutes',
                                    ]
                                    
                
                train_df['satisfaction'] = train_df['satisfaction'].map({'neutral or dissatisfied':0 , 'satisfied':1})
                test_df['satisfaction'] = test_df['satisfaction'].map({'neutral or dissatisfied':0 , 'satisfied':1})

                input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
                target_feature_train_df=train_df[target_column_name]

                input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df=test_df[target_column_name]

                logging.info(
                    f"Applying preprocessing object on training dataframe and testing dataframe."
                )
               

                input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

                # print(input_feature_train_arr)
                # input()


                train_arr = np.c_[
                    input_feature_train_arr, np.array(target_feature_train_df)
                ]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

                logging.info(f"Saved preprocessing object.")

                save_object(

                    file_path=self.data_transformation_config.preprocessor_obj_file_path,
                    obj=preprocessing_obj

                )

                return (
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_obj_file_path,
                )
            except Exception as e:
                raise CustomException(e,sys)