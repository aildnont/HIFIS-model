import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import yaml
import os

def load_df(path):
    '''
    Load a Pandas dataframe from a CSV file
    :param path: The file path of the CSV file
    :return: A Pandas dataframe
    '''
    # Read HIFIS data into a Pandas dataframe
    df = pd.read_csv(path, encoding="ISO-8859-1", low_memory=False)

    # Delete the first row of "-------" entries, as it is an artifact of the script used to pull down the database
    df.drop(df.index[0], inplace=True)
    return df

def cleanse_categorical_features(df, categorical_features):
    '''
    For all categorical features, replace entries containing unique values with "Other"
    :param df: a Pandas dataframe
    :param categorical_features: list of categorical features in the dataframe
    :return: the updated dataframe
    '''
    for feature in categorical_features:
        counts = df[feature].value_counts() # Get a Pandas series of all the values for
        single_values = counts[counts == 1] # Get values for this categorical feature unique to 1 row
        df[feature].loc[(df[feature].isin(list(single_values.index)))] = "Other" # Replace unique instances of categorical values with "Other"
    return df

def classify_cat_features(df, cat_features):
    '''
    Classify categorical features as either single- or multi-valued.
    :param df: Pandas dataframe
    :param cat_features: List of categorical features
    :return: list of single-valued categorical features, list of multi-valued categorical features
    '''
    def classify_features(client_df):
        '''
        Helper function for categorical feature classification, distributed across clients.
        :param client_df: Dataframe with 1 client's records
        '''
        for feature in cat_features:
            # If this feature takes more than 1 value per client, move it to the list of multi-valued features
            if client_df[feature].nunique() > 1:
                sv_cat_features.remove(feature)
                mv_cat_features.append(feature)
                return

    sv_cat_features = cat_features  # First, assume all categorical features are single-valued
    mv_cat_features = []
    df.groupby('ClientID').progress_apply(classify_features)
    return sv_cat_features, mv_cat_features

def vec_multi_value_cat_features(df, mv_cat_features):
    '''
        Converts multi-valued categorical features to vectorized format and appends to the dataframe
        :param df: A Pandas dataframe
        :param mv_categorical_features: The names of the categorical features to vectorize
        :return: dataframe containing vectorized features, list of vectorized feature names
        '''
    mv_vec_categorical_features = []
    for feature in mv_cat_features:
        df_temp = pd.get_dummies(df[feature], prefix=feature)  # Create temporary dataframe of this feature vectorized
        df = pd.concat((df, df_temp), axis=1)  # Concatenate temp one hot dataframe with original dataframe
        df = df.drop(feature, axis=1)  # Drop the original feature
        vectorized_headers_list = list(df_temp)
        for i in range(len(vectorized_headers_list)):
            mv_vec_categorical_features.append(vectorized_headers_list[i])  # Keep track of vectorized features
    return df, mv_vec_categorical_features

def vec_single_value_cat_features(df, sv_cat_features):
    '''
    Converts single-valued categorical features to one-hot encoded format (i.e. vectorization) and appends to the dataframe.
    Keeps track of a mapping from feature indices to categorical values, for interpretability purposes.
    :param df: A Pandas dataframe
    :param sv_cat_features: The names of the categorical features to encode
    :return: dataframe containing one-hot encoded features, list of one-hot encoded feature names
    '''
    vec_cat_features = []
    cat_feature_idxs = [df.columns.get_loc(c) for c in sv_cat_features if c in df]   # List of categorical column indices
    cat_value_names = {}    # Dictionary of categorical feature indices and corresponding names of feature values
    for i in range(len(sv_cat_features)):
        df_temp = pd.get_dummies(df[sv_cat_features[i]], prefix=sv_cat_features[i])  # Create temporary dataframe of this feature vectorized
        value_names = [name[(name.index('_') + 1):] for name in df_temp.columns]    # Get list of values of this
        cat_value_names[cat_feature_idxs[i]] = value_names
        df = pd.concat((df, df_temp), axis=1)  # Concatenate temp one hot dataframe with original dataframe
        df = df.drop(sv_cat_features[i], axis=1)  # Drop the original feature
        vec_cat_features.extend(df_temp.columns)  # Keep track of vectorized feature columns
    return df, vec_cat_features, cat_feature_idxs, cat_value_names

def process_timestamps(df):
    '''
    Convert timestamps in raw date to datetimes
    :param df: A Pandas dataframe
    :return: The dataframe with its datetime fields updated accordingly
    '''
    features_list = list(df)  # Get a list of features
    for feature in features_list:
        if ('Date' in feature) or ('Start' in feature) or ('End' in feature):
            df[feature] = pd.to_datetime(df[feature], infer_datetime_format=True, errors='coerce')
    return df

def remove_n_weeks(df, n_weeks, gt_end_date, dated_feats):
    '''
    Remove records from the dataframe that have timestamps in the n weeks leading up to the ground truth date
    :param df: Pandas dataframe
    :param n_weeks: number of recent weeks to remove records
    :param gt_end_date: the date used for ground truth calculation
    :return: updated dataframe with the relevant rows removed
    '''
    train_end_date = gt_end_date - timedelta(days=(n_weeks * 7))    # Maximum for training set records
    print("Train end date: ", train_end_date)
    df = df[df['ServiceStartDate'] <= train_end_date]               # Delete rows where service occurred after this date
    df['ServiceEndDate'] = df['ServiceEndDate'].clip(upper=train_end_date)  # Set end date for ongoing services to this date

    # Set features with dated events occurring after the maximum training set date to null
    for feat in dated_feats:
        idxs_to_update = df[df[feat] > train_end_date].index.tolist()
        df.loc[idxs_to_update, dated_feats[feat]] = np.nan
    return df

def convert_yn_to_boolean(df, categorical_features, noncategorical_features):
    '''
    Convert yes/no features to boolean features. Avoids vectorization.
    :param df: a Pandas dataframe
    :return: updated dataframe with the yes/no features converted to boolean values
    '''
    yn_features = []    # List of Yes/No feature names
    for feature in categorical_features:
        if "YN" in feature:
            new_feature = feature[0:feature.index('YN')]
            df[new_feature] = np.where(df[feature] == 'Y', 1, 0)
            df.drop(feature, axis=1, inplace=True)
            yn_features.append(feature)
            noncategorical_features.append(new_feature)
    # Remove Y/N features from list of categorical features
    categorical_features = [f for f in categorical_features if f not in yn_features]
    return df, categorical_features, noncategorical_features

def calculate_ground_truth(df, chronic_threshold, days, end_date):
    '''
    Iterate through dataset by client to calculate ground truth
    :param df: a Pandas dataframe
    :param chronic_threshold: Minimum # of days spent in shelter to be considered chronically homeless
    :param days: Number of days over which to cound # days spent in shelter
    :param end_date: The last date of the time period to consider
    :return: the dataframe with ground truth appended at the end
    '''

    def client_gt(client_df):
        '''
        Helper function ground truth calculation.
        :param client_df: A dataframe containing all rows for a client
        :return: the client dataframe ground truth calculated correctly
        '''
        client_df.sort_values(by=['ServiceStartDate'], inplace=True) # Sort records by service start date
        gt_stays = 0 # Keep track of total stays, as well as # stays during ground truth time range
        last_end = pd.to_datetime(0)
        last_start = pd.to_datetime(0)

        # Iterate over all of client's records. Note itertuples() is faster than iterrows().
        for row in client_df.itertuples():
            stay_start = getattr(row, 'ServiceStartDate')
            stay_end = min(getattr(row, 'ServiceEndDate'), end_date) # If stay is ongoing through end_date, set end of stay as end_date
            if stay_start != last_start:
                if (stay_start.date() >= start_date.date()) or (stay_end.date() >= start_date.date()):
                    gt_stay_start = max(start_date, stay_start) # Account for cases where stay start earlier than start of range
                    gt_stays += (stay_end - gt_stay_start).days + (gt_stay_start.date() != last_end.date())
                last_end = stay_end
                last_start = stay_start

        # Determine if client meets ground truth threshold
        if gt_stays >= chronic_threshold:
            client_df['GroundTruth'] = 1
            stats['num_pos'] += 1
        else:
            stats['num_neg'] += 1
        return client_df

    start_date = end_date - timedelta(days=days) # Get start of ground truth window
    stats = {'num_neg': 0, 'num_pos': 0}  # Record number of clients in each class
    df['GroundTruth'] = 0
    df_temp = df.loc[(df['ServiceType'] == 'Stay')]
    df_temp = df_temp.groupby('ClientID').progress_apply(client_gt)
    df_temp = df_temp.droplevel('ClientID', axis='index')
    df.update(df_temp)  # Update all rows with corresponding stay length and ground truth
    return df, stats['num_pos'], stats['num_neg']

def calculate_client_features(df, end_date, counted_services):
    '''
    Iterate through dataset by client to calculate some features (total stays, monthly income)
    :param df: a Pandas dataframe
    :param end_date: The last date of the time period to consider
    :return: the dataframe with the new features and ground truth appended at the end
    '''

    def client_features(client_df):
        '''
        Helper function for total stay, total income and ground truth calculation.
        To be used on a subset of the dataframe
        :param client_df: A dataframe containing all rows for a client
        :return: the client dataframe with total stays and ground truth columns appended
        '''
        client_df.sort_values(by=['ServiceStartDate'], inplace=True) # Sort records by service start date
        total_stays = 0 # Keep track of total stays, as well as # stays during ground truth time range
        last_stay_end = pd.to_datetime(0)
        last_stay_start = pd.to_datetime(0)
        last_service_end = pd.to_datetime(0)
        last_service = ''

        # Iterate over all of client's records. Note itertuples() is faster than iterrows().
        for row in client_df.itertuples():
            service_start = getattr(row, 'ServiceStartDate')
            service_end = min(getattr(row, 'ServiceEndDate'), end_date) # If stay is ongoing through end_date, set end of stay as end_date
            if (getattr(row, 'ServiceType') == 'Stay'):
                if (service_start != last_stay_start):
                    total_stays += (service_end.date() - service_start.date()).days + (service_start.date() != last_stay_end.date())
                    last_stay_end = service_end
                    last_stay_start = service_start
            elif (service_end != last_service_end) or (getattr(row, 'ServiceType') != last_service):
                service = getattr(row, 'ServiceType')
                client_df['Num_' + service] += 1    # Increment # of times this service was accessed by this client
                last_service_end = service_end
                last_service = service
        client_df['TotalStays'] = total_stays

        # Calculate total monthly income for client
        client_income_df = client_df.drop_duplicates(subset=['IncomeType'])
        client_df['IncomeTotal'] = client_income_df['MonthlyAmount'].sum()
        return client_df

    df['TotalStays'] = 0    # Create columns for stays and income total
    df['IncomeTotal'] = 0
    df['MonthlyAmount'] = pd.to_numeric(df['MonthlyAmount'])
    numerical_service_features = []
    for service in counted_services:
        df['Num_' + service] = 0
        numerical_service_features.append('Num_' + service)
    df_temp = df.copy()
    df_temp = df_temp.groupby('ClientID').progress_apply(client_features)
    df_temp = df_temp.droplevel('ClientID', axis='index')
    df.update(df_temp)  # Update all rows with corresponding stay length and total income
    return df, numerical_service_features

def aggregate_df(df, noncategorical_features, vec_mv_categorical_features, vec_sv_categorical_features, numerical_service_features):
    '''
    Build a dictionary of columns and arguments to feed into the aggregation function, and aggregate the dataframe
    :param df: a Pandas dataframe
    :param noncategorical_features: list of noncategorical features
    :param ohe_features: list of one-hot encoded features
    :return: A grouped dataframe with one row for each client
    '''
    grouping_dictionary = {}
    temp_dict = {}
    non_categorical_features = noncategorical_features
    non_categorical_features.remove('ServiceStartDate')
    non_categorical_features.remove('ClientID')

    # Create a dictionary of column names and function names to pass into the groupby function
    for i in range(len(non_categorical_features)):
        grouping_dictionary[non_categorical_features[i]] = 'first'  # Group noncategorical features by first occurrence
    for i in range(len(vec_sv_categorical_features)):
        temp_dict[vec_sv_categorical_features[i]] = 'first'  # Group one hot features by max value
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {}
    for i in range(len(vec_mv_categorical_features)):
        temp_dict[vec_mv_categorical_features[i]] = 'max'  # Group single categorical features by first value
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {}
    for i in range(len(numerical_service_features)):
        temp_dict[numerical_service_features[i]] = 'first'  # Group one hot features by max value
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {'IncomeTotal': 'first', 'TotalStays': 'max',}
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    if 'GroundTruth' in df.columns:
        temp_dict = {'GroundTruth': 'max', }
        grouping_dictionary = {**grouping_dictionary, **temp_dict}

    # Group the data by ClientID using the dictionary created above
    df_unique_clients = df.groupby(['ClientID']).agg(grouping_dictionary)
    return df_unique_clients


def preprocess(n_weeks=None, load_gt=False, classify_cat_feats=True):
    '''
    Load results of the HIFIS SQL query and process the data into features for training a model.
    :param n_weeks: Prediction horizon
    '''
    run_start = datetime.today()
    tqdm.pandas()
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    config = yaml.full_load(input_stream)       # Load config data
    categorical_features = config['DATA']['CATEGORICAL_FEATURES']
    noncategorical_features = config['DATA']['NONCATEGORICAL_FEATURES']
    features_to_drop_last = config['DATA']['FEATURES_TO_DROP_LAST']
    GROUND_TRUTH_DURATION = 365     # In days. Set to 1 year.
    if n_weeks is None:
        N_WEEKS = config['DATA']['N_WEEKS']
    else:
        N_WEEKS = n_weeks

    # Load HIFIS database into Pandas dataframe
    print("Loading HIFIS data.")
    df = load_df(config['PATHS']['RAW_DATA'])

    # Delete unwanted columns
    print("Dropping some features.")
    for feature in config['DATA']['FEATURES_TO_DROP_FIRST']:
        df.drop(feature, axis=1, inplace=True)

    # Create a new boolean feature that indicates whether client has family
    df['HasFamily'] = np.where((~df['FamilyID'].isnull()), 1, 0)
    noncategorical_features.append('HasFamily')
    noncategorical_features.remove('FamilyID')

    # Convert yes/no features to boolean features
    print("Convert yes/no categorical features to boolean")
    df, categorical_features, noncategorical_features = convert_yn_to_boolean(df, categorical_features, noncategorical_features)

    # Replace null ServiceEndDate entries with today's date. Assumes client is receiving ongoing services.
    df['ServiceEndDate'] = np.where(df['ServiceEndDate'].isnull(), pd.to_datetime('today'), df['ServiceEndDate'])

    # Convert all timestamps to datetime objects
    print("Converting timestamps to datetimes.")
    df = process_timestamps(df)

    # Calculate ground truth
    gt_end_date = pd.to_datetime(config['DATA']['GROUND_TRUTH_DATE'])
    if not load_gt:
        print("Calculating ground truth.")
        df, num_pos, num_neg = calculate_ground_truth(df, config['DATA']['CHRONIC_THRESHOLD'], GROUND_TRUTH_DURATION, gt_end_date)

    # Remove records from the database from n weeks ago and onwards
    print("Removing records ", N_WEEKS, " weeks back.")
    df = remove_n_weeks(df, N_WEEKS, gt_end_date, config['DATA']['OTHER_TIMED_FEATURES'])

    # Compute total stays, total monthly income, total # services accessed for each client.
    print("Calculating total stays, monthly income.")
    df, numerical_service_features = calculate_client_features(df, gt_end_date, config['DATA']['COUNTED_SERVICE_FEATURES'])

    # Index dataframe by the service start column
    df = df.set_index('ServiceStartDate')

    # Create an "Other" value for each categorical variable, which will serve as the value for unique entries
    print("Cleansing categorical features of unique values.")
    df = cleanse_categorical_features(df, categorical_features)

    print("Separating multi and single-valued categorical features.")
    if classify_cat_feats:
        sv_cat_features, mv_cat_features = classify_cat_features(df, categorical_features)
    else:
        input_stream = open(os.getcwd() + config['PATHS']['INTERPRETABILITY'], 'r')
        cfg_gen = yaml.full_load(input_stream)  # Get config data generated from previous preprocessing
        sv_cat_features = cfg_gen['SV_CAT_FEATURES']
        mv_cat_features = cfg_gen['MV_CAT_FEATURES']

    # Vectorize the multi-valued categorical features
    print("Vectorizing multi-valued categorical features.")
    df, vec_mv_cat_features = vec_multi_value_cat_features(df, mv_cat_features)

    # Amalgamate rows to have one entry per client
    print("Grouping the dataframe.")
    df_clients = aggregate_df(df, noncategorical_features, vec_mv_cat_features, sv_cat_features, numerical_service_features)

    # Drop unnecessary features
    print("Dropping unnecessary features.")
    for column in features_to_drop_last:
        if column in df_clients.columns:
            df_clients.drop(column, axis=1, inplace=True)

    # Get list of remaining noncategorical features
    noncat_feats_gone = [f for f in noncategorical_features if f not in df_clients.columns]
    for feature in noncat_feats_gone:
        noncategorical_features.remove(feature)
    noncategorical_features.extend(numerical_service_features)
    noncategorical_features.extend(['IncomeTotal', 'TotalStays'])

    # Replace all instances of NaN in the dataframe with 0 or "Unknown"
    df_clients[sv_cat_features] = df_clients[sv_cat_features].fillna("Unknown")
    df_clients[noncategorical_features] = df_clients[noncategorical_features].fillna(0)

    # Load ground truth if not calculated already. Otherwise, save ground truth.
    if load_gt:
        df_gt = load_df(config['PATHS']['GROUND_TRUTH'])    # Load ground truth from file
        df_gt = df_gt.set_index('ClientID')
        df_gt.index = df_gt.index.astype(int)
        df_clients.index = df_clients.index.astype(int)
        df_clients = df_clients.join(df_gt) # Set ground truth for all clients to their saved values
        df_clients['GroundTruth'] = df_clients['GroundTruth'].fillna(0)
    else:
        df_gt = df_clients['GroundTruth']
        df_gt.to_csv(config['PATHS']['GROUND_TRUTH'], sep=',', header=True)    # Save ground truth

    # Vectorize single-valued categorical features. Keep track of feature names and values.
    print("Vectorizing single-valued categorical features.")
    df_ohe_clients, vec_sv_cat_features, sv_cat_feature_idxs, sv_cat_values = vec_single_value_cat_features(df_clients, sv_cat_features)

    # Save processed dataset
    print("Saving data.")
    df_clients.to_csv(config['PATHS']['PROCESSED_DATA'], sep=',', header=True)
    df_ohe_clients.to_csv(config['PATHS']['PROCESSED_OHE_DATA'], sep=',', header=True)

    # For producing interpretable results with categorical data:
    interpretability_info = {}
    interpretability_info['SV_CAT_FEATURES'] = sv_cat_features
    interpretability_info['MV_CAT_FEATURES'] = mv_cat_features
    interpretability_info['VEC_SV_CAT_FEATURES'] = vec_sv_cat_features
    interpretability_info['SV_CAT_FEATURE_IDXS'] = sv_cat_feature_idxs
    interpretability_info['SV_CAT_VALUES'] = sv_cat_values
    interpretability_info['NON_CAT_FEATURES'] = noncategorical_features
    with open(config['PATHS']['INTERPRETABILITY'], 'w') as file:
        interpretability_doc = yaml.dump(interpretability_info, file)

    # Log some useful stats
    if not load_gt:
        print("# clients in last year meeting homelessness criteria = ", num_pos)
        print("# clients in last year meeting homelessness criteria = ", num_neg)
        print("% positive for chronic homelessness = ", 100 * num_pos / (num_pos + num_neg))
        print("Runtime = ", ((datetime.today() - run_start).seconds / 60), " min")

if __name__ == '__main__':
    preprocess(n_weeks=16, load_gt=True, classify_cat_feats=True)


