import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import os, sys
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logger_utility import logger

def load_params(params_path : str) -> float:
    logger.info("Loading params")
    config = yaml.safe_load(open(params_path, 'r'))
    test_size = config['data_ingestion']['test_size']
    return test_size

def read_data(url: str):  
    try:
        logger.info("Reading data from url")
        df = pd.read_csv(url)
        return df
    except Exception as e:
        print(e)
        raise

def process_data(df) -> pd.DataFrame:
    df.drop(columns=['tweet_id'],inplace=True)
    final_df = df[df['sentiment'].isin(['happiness','sadness'])]

    final_df['sentiment'].replace({'happiness':1, 'sadness':0},inplace=True)

    return final_df

def save_data(df):
    logger.info("Saving the data")
    test_size = load_params("params.yaml")
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    data_path = os.path.join("data", "raw")

    os.makedirs(data_path, exist_ok=True)

    train_data.to_csv(os.path.join(data_path, "train.csv"))
    test_data.to_csv(os.path.join(data_path, "test.csv"))
    logger.info("Data saved")

def main(): 
    url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
    df = read_data(url)
    final_df = process_data(df)
    save_data(final_df)

if __name__ == "__main__":
    main()

