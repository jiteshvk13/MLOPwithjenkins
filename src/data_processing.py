import pandas as pd
from config.paths_config import *
from src.logger import get_logger
from src.custom_exception import CustomException
import sys

logger = get_logger(__name__)

class DataProcessor:
    def __init__(self):
        self.train_path = TRAIN_DATA_PATH
        self.processed_data_path = PROCESSED_DATA_PATH

    def load_data(self):
        try:
            logger.info("Data Processing Started")
            df = pd.read_csv(self.train_path)
            logger.info("Data Read Successfully")
            return df
        except Exception as e:
            logger.error("Problem While Loading Data")
            raise CustomException("Error While loading Data")
        
    def drop_unnecesary_columns(self, df, columns):
        try:
            logger.info(f"dropping Unnecessary Columns : {columns}") 
            df = df.drop(columns = columns ,axis=1)
            logger.info("Columns Dropped Succesfullly")
            return df
        except Exception as e:
            logger.error("Problem While Dropping Columns")
            raise CustomException("Error While Dropping Columns")

    
    def handle_outliers(self, df, columns):
        try:
            logger.info(f"Handling Outliers : Columns = {columns}")
            for column in columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
            
            logger.info("Outlier handled Succesfully")
            return df
        
        except Exception as e:
            logger.error("Problem While Dropping Columns")
            raise CustomException("Error While Dropping Columns")  

    def handle_null_values(self,df,columns):
        try:
            logger.info("handling Null Values")
            df[columns] = df[columns].fillna(df[columns].median())
            logger.info("Missing Values Handles Succesfully")
            return df
        except Exception as e:
            logger.error("Problem Handling null Values")
            raise CustomException("Error Handling null Values")  

    def save_data(self, df):
        try:
            os.makedirs(PROCESSED_DIR, exist_ok = True)
            df.to_csv(self.processed_data_path, index = False)
            logger.info("Processed Data Saved Successfully")

        except Exception as e:
            logger.error("Problem while Saving the Data")
            raise CustomException("Error while Saving the Data")       
        
    def run(self):
        try: 
            logger.info("Starting the Data Preprocessing")
            df = self.load_data()
            df = self.drop_unnecesary_columns(df,["MyUnknownColumn", "id"])
            columns_to_handle= ['Flight Distance','Departure Delay in Minutes','Arrival Delay in Minutes', 'Checkin service']

            df = self.handle_outliers(df,columns_to_handle)
            df = self.handle_null_values(df, "Arrival Delay in Minutes")

            self.save_data(df)

            logger.info("Data Processing Pipeline Completed Successfully")
        
        except Exception as e:
            logger.error("Error while Creating Pipeline")



if  __name__ == "__main__":
    processor = DataProcessor()
    processor.run()