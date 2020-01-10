import numpy as np
import pandas as pd
from datetime import datetime, timedelta
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

def ohe_categorical_features(df, categorical_features):
    '''
    Converts categorical features to one-hot encoded format (i.e. vectorization) and appends to the dataframe
    :param df: A Pandas dataframe
    :param categorical_features: The names of the categorical features to encode
    :return: dataframe containing one-hot encoded features, list of one-hot encoded feature names
    '''
    ohe_categorical_features = []
    for feature in categorical_features:
        df_temp = pd.get_dummies(df[feature], prefix=feature)  # Create temporary dataframe of this feature one-hot encoded
        df = pd.concat((df, df_temp), axis=1)  # Concatenate temp one hot dataframe with original dataframe
        df = df.drop(feature, axis=1)  # Drop the original feature
        vectorized_headers_list = list(df_temp)
        for i in range(len(vectorized_headers_list)):
            ohe_categorical_features.append(vectorized_headers_list[i]) # Keep track of one-hot encoded features
    return df, ohe_categorical_features


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

def make_start_end_features(df, noncategorical_features, timed_service_features, features_to_drop_last):
    '''
    Create and add features for start and end times of certain features
    :param df: a Pandas dataframe
    :param noncategorical_features: list of noncategorical features in the dataset
    :param timed_service_features: list of service features that occur at events spaced in time
    :return: the updated dataframe, the updated list of noncategorical features
    '''
    # Replace null ServiceEndDate entries with today's date. Assumes client is receiving ongoing services.
    df['ServiceEndDate'] = np.where(df['ServiceEndDate'].isnull(), pd.to_datetime('today'), df['ServiceEndDate'])

    # Create a new feature for the start and end date of each service feature
    for feature in timed_service_features:
        feature_start_name = feature + 'StartDate'
        feature_end_name = feature + 'EndDate'
        df[feature_start_name] = np.where(df['ServiceType'] == feature, df['ServiceStartDate'], 0)
        df[feature_end_name] = np.where(df['ServiceType'] == feature, df['ServiceEndDate'], 0)
        noncategorical_features.extend([feature_start_name, feature_end_name])
        features_to_drop_last.extend([feature_start_name, feature_end_name])
    return df, noncategorical_features, features_to_drop_last

def calculate_length_features(df, timed_service_features):
    '''
    Create features for total length of time for features with start and end dates
    :param df: a Pandas dataframe
    :param timed_service_features: Service features whose start and end dates were calculated in this script
    :return:
    '''
    seconds_per_day = 60 * 60 * 24 # 60sec/min * 60min/hr * 24hr/day
    length_features = [] # Keep track of features identifying a time duration
    for feature in timed_service_features:
        length_feature_name = 'LengthOf' + feature + 'Days'
        df[length_feature_name] = (df[feature + 'EndDate'] - df[feature + 'StartDate']).dt.total_seconds() / seconds_per_day
        length_features.append(length_feature_name)
    return df, length_features

def create_service_num_features(df, counted_service_features):
    '''
    Creates boolean feature indicating a service record for all counted service features.
    The count will be determined later in the aggregate_df function, when these columns are aggregated by sum
    :param df: a Pandas dataframe
    :param counted_service_features: a list service features whose occurrences will be summed for each client
    :return: updated dataframe, list of feature names corresponding to counts of each service
    '''
    numerical_service_features = []
    for feature in counted_service_features:
        new_feature_name = "Num_" + feature
        df[new_feature_name] = np.where((df['ServiceType'] == feature), 1, 0) # represents presence of a service record
        numerical_service_features.append(new_feature_name)
    return df, numerical_service_features

def convert_yn_to_boolean(df, categorical_features, noncategorical_features):
    '''
    Convert yes/no features to boolean features. Avoids vectorization.
    :param df: a Pandas dataframe
    :return: updated dataframe with the yes/no features converted to boolean values
    '''
    for feature_name in categorical_features:
        if "YN" in feature_name:
            new_feature_name = feature_name[0:feature_name.index('YN')]
            df[new_feature_name] = np.where(df[feature_name] == 'Y', 1, 0)
            df.drop(feature_name, axis=1, inplace=True)
            categorical_features.remove(feature_name)
            noncategorical_features.append(new_feature_name)
    return df, categorical_features, noncategorical_features

def set_ground_truths(df, chronic_threshold, days, end_date):
    '''
    Determine ground truth for each client, which is defined as a certain number of days spent in a shelter
    :param df: a Pandas dataframe
    :param chronic_threshold: Minimum # of days spent in shelter to be considered chronically homeless
    :param days: Number of days over which to cound # days spent in shelter
    :param end_date: The last date of the time period to consider
    :return: the dataframe with a ground truth column appended at the end
    '''
    def calculate_stays(client_df):
        client_df.sort_values(by=['ServiceStartDate'], inplace=True)
        total_stays = gt_stays = 0
        last_end = pd.to_datetime(0)
        last_start = pd.to_datetime(0)
        for row in client_df.itertuples():
            stay_start = getattr(row, 'StayStartDate')
            stay_end = getattr(row, 'StayEndDate')
            if stay_start != last_start:
                total_stays += (stay_end.date() - stay_start.date()).days
                if stay_start > start_date:
                    gt_stays += (stay_end - stay_start).days
                    if (stay_start - last_end).days != 0:
                        gt_stays += 1
                if (stay_start - last_end).days != 0:
                    total_stays += 1
                last_end = stay_end
                last_start = stay_start
        client_df['TotalStays'] = total_stays
        if gt_stays >= chronic_threshold:
            client_df['GroundTruth'] = 1
            stats['num_pos'] += 1
        else:
            stats['num_neg'] += 1
        return client_df
    start_date = end_date - timedelta(days=days)
    stats = {'num_neg':0, 'num_pos':0}
    df['TotalStays'] = 0
    df['GroundTruth'] = 0
    df_temp = df.loc[(df['ServiceType'] == 'Stay') & (df['ServiceEndDate'] <= end_date)]
    df_temp = df_temp.groupby('ClientID').apply(calculate_stays)
    df_temp = df_temp.droplevel('ClientID', axis='index')
    df.update(df_temp)
    return df, stats['num_pos'], stats['num_neg']

def aggregate_df(df, noncategorical_features, ohe_categorical_features, numerical_service_features):
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
    for i in range(len(ohe_categorical_features)):
        temp_dict[ohe_categorical_features[i]] = 'max'  # Group one hot features by max value
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {}
    for i in range(len(numerical_service_features)):
        temp_dict[numerical_service_features[i]] = 'sum'  # Group one hot features by max value
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {}
    for i in range(len(length_features)):
        temp_dict[length_features[i]] = 'sum'
    grouping_dictionary = {**grouping_dictionary, **temp_dict}
    temp_dict = {'IncomeTotal': 'first', 'TotalStays': 'max', 'GroundTruth': 'max', }
    grouping_dictionary = {**grouping_dictionary, **temp_dict}

    # Group the data by ClientID using the dictionary created above
    df_unique_clients = df.groupby(['ClientID']).agg(grouping_dictionary)
    return df_unique_clients


# Load config data
input_stream = open(os.getcwd() + "/config.yml", 'r')
config = yaml.full_load(input_stream)
categorical_features = config['DATA']['CATEGORICAL_FEATURES']
noncategorical_features = config['DATA']['NONCATEGORICAL_FEATURES']
features_to_drop_last = config['DATA']['FEATURES_TO_DROP_LAST']
GROUND_TRUTH_DURATION = 365     # In days. Set to 1 year.

# Load HIFIS database into Pandas dataframe
print("Loading HIFIS data.")
df = load_df(config['PATHS']['RAW_DATA'])

# Delete unwanted columns
print("Dropping some features.")
for feature in config['DATA']['FEATURES_TO_DROP_FIRST']:
    df.drop(feature, axis=1, inplace=True)

# Create a new feature for start and end of each service type that is accessed over multiple timeframes
print("Adding features for service start/end dates, family.")
df, noncategorical_features, features_to_drop_last = make_start_end_features(df, noncategorical_features, config['DATA']['TIMED_SERVICE_FEATURES'], features_to_drop_last)

# Create a new boolean feature that indicates whether client has family
df['HasFamily'] = np.where(df['FamilyID'] != 'NULL', 1, 0)
noncategorical_features.append('HasFamily')

# Convert yes/no features to boolean features
print("Convert yes/no categorical features to boolean")
df, categorical_features, noncategorical_features = convert_yn_to_boolean(df, categorical_features, noncategorical_features)

# Convert all timestamps to datetime objects
print("Converting timestamps to datetimes.")
df = process_timestamps(df)

# Create length of service feature to describe the duration of timestamped features
print("Calculating length features.")
df, length_features = calculate_length_features(df, config['DATA']['TIMED_SERVICE_FEATURES'])

# Add number of food bank trips as a feature
print("Create features for service types that will be summed")
df, numerical_service_features = create_service_num_features(df, config['DATA']['COUNTED_SERVICE_FEATURES'])

# Compute ground truth for each client. Ground truth
print("Calculating ground truths.")
gt_end_date = pd.to_datetime(config['DATA']['GROUND_TRUTH_DATE'])
df, num_pos, num_neg = set_ground_truths(df, config['DATA']['CHRONIC_THRESHOLD'], GROUND_TRUTH_DURATION, gt_end_date)

# Index dataframe by the service start column
df = df.set_index('ServiceStartDate')

# Get total monthly incomes for each client and add as feature
print("Adding total monthly income as feature.")
df['MonthlyAmount'] = pd.to_numeric(df['MonthlyAmount'])
df['IncomeTotal'] = df.groupby(['ServiceID', 'ClientID', 'ServiceStartDate'])['MonthlyAmount'].transform('sum')

# Drop duplicate rows for ClientID and ServiceID. Should now have 1 row for each client.
df.drop_duplicates(subset=['ServiceID', 'ClientID'], keep='first', inplace=True)

# Log some useful stats
print("# clients in last year meeting homeslessness criteria = ", num_pos)
print("# clients in last year meeting homeslessness criteria = ", num_neg)
print("% positive for chronic homelessness = ", 100 * num_pos / (num_pos + num_neg))

# Amalgamate rows to have one entry per client
print("Grouping the dataframe.")
df_unique_clients = aggregate_df(df, noncategorical_features, categorical_features, numerical_service_features)

# Create an "Other" value for each categorical variable, which will serve as the value for unique entries
print("Cleansing categorical features of unique values.")
df_unique_clients = cleanse_categorical_features(df_unique_clients, categorical_features)

# One hot encode the categorical features
print("One-hot encoding categorical features.")
df_unique_clients, ohe_categorical_features = ohe_categorical_features(df_unique_clients, categorical_features)

# Drop unnecessary features
print("Dropping unnecessary features.")
for column in features_to_drop_last:
    df_unique_clients.drop(column, axis=1, inplace=True)

# Replace all instances of NaN in the dataframe with 0
df_unique_clients.fillna(0, inplace=True)

# Save vectorized data
print("Saving data.")
df_unique_clients.to_csv(config['PATHS']['VECTORIZED_DATA'], sep=',', header=True)


