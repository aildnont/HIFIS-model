import json
import os
import yaml
import pandas as pd
from tqdm import tqdm

def single_client_record(client_df):
    client_answers = dict.fromkeys(questions, [None])
    for row in client_df.itertuples():
        client_answers[getattr(row, 'QuestionE')] = [getattr(row, 'ScoreValue')]
    return pd.DataFrame.from_dict(client_answers)


cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))       # Load project config

# Read JSON file containing SPDAT information. Remove line breaks.
with open('data/raw/SPDATS.json', 'rb') as f:
    json_str = f.read().decode('utf-16').replace('\r', '').replace('\n', '')
spdats = json.loads(json_str)['VISPDATS']   # Convert to object and get the list of SPDATs
df = pd.DataFrame(spdats)

tqdm.pandas()
questions = df['QuestionE'].unique()    # Get list of unique questions across all SPDAT versions
df_clients = df.groupby('ClientID').progress_apply(single_client_record)
df_clients.columns = df_clients.columns.str.replace('%', '\%')      # Remove bad characters that prevents debug inspection
df_clients = df_clients.droplevel(level=1, axis='index')

