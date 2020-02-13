import json
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_spdat_feat_types(df):
    '''
    Get lists of features containing the single-valued categorical and noncategorical feature names in the SPDAT data
    :param df: Pandas DataFrame containing client SPDAT question info
    :return: List of single-valued categorical features, list of noncategorical features
    '''
    sv_cat_feats = []
    noncat_feats = []
    for column in df.columns:
        if df[column].dtype == 'object':
            sv_cat_feats.append(column)
        else:
            noncat_feats.append(column)
    return sv_cat_feats, noncat_feats


def get_spdat_data(spdat_path):
    '''
    Read SPDAT data from raw SPDAT file and output a dataframe containing clients' answers to the questions.
    :param spdat_path: The file path of the raw SPDAT data
    :return: A DataFrame in which each row is a client's answers to SPDAT questions
    '''

    def single_client_record(client_df):
        '''
        Helper function for SPDAT data preprocessing. Processes records for a single client.
        :param client_df: Raw SPDAT data for 1 client
        :return: DataFrame with 1 row detailing client's answers to SPDAT questions
        '''
        client_answers = dict.fromkeys(questions, [np.nan])   # The columns will be SPDAT questions
        for row in client_df.itertuples():
            answer = str(getattr(row, 'ScoreValue'))
            answer = float(answer) if answer.isnumeric() else answer
            client_answers[getattr(row, 'QuestionE')] = [answer]    # Set values to client's answers
        dfdd = pd.DataFrame.from_dict(client_answers)
        dfdd.columns = dfdd.columns.str.replace('%', '')
        return pd.DataFrame.from_dict(client_answers)

    # Read JSON file containing SPDAT information. Remove line breaks.
    with open(spdat_path, 'rb') as f:
        json_str = f.read().decode('utf-16').replace('\r', '').replace('\n', '')
    spdats = json.loads(json_str)['VISPDATS']   # Convert to object and get the list of SPDATs
    df = pd.DataFrame(spdats)                   # Convert JSON object to pandas DataFrame
    df.fillna(0, inplace=True)

    tqdm.pandas()
    questions = df['QuestionE'].unique()    # Get list of unique questions across all SPDAT versions

    # Build a DataFrame in which each row is a client's answer to SPDAT questions
    df_clients = df.groupby('ClientID').progress_apply(single_client_record)
    df_clients.columns = df_clients.columns.str.replace('%', '')      # Remove bad characters that prevent debug inspection
    df_clients = df_clients.droplevel(level=1, axis='index')            # Ensure index is ClientID
    print("# of clients with SPDAT = " + str(df_clients.shape[0]))

    sv_cat_feats, noncat_feats = get_spdat_feat_types(df_clients)
    return df_clients, sv_cat_feats, noncat_feats
