import pandas as pd
import yaml
import os
import datetime
import numpy as np
from sklearn.externals.joblib import load
from tensorflow.keras.models import load_model
from src.data.preprocess import preprocess

def predict(data_path=None):

    # Load project config data
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)

    # Preprocess the data, using pre-existing sklearn transformers and classification of categorical features.
    data = preprocess(n_weeks=12, include_gt=False, calculate_gt=False, classify_cat_feats=False, load_ct=True,
                      data_path=data_path)



if __name__ == '__main__':
    predict(data_path='data/raw/HIFIS_Clients_test.csv')