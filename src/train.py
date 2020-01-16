import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from src.models.models import model1

# Load project config data
input_stream = open(os.getcwd() + "/config.yml", 'r')
cfg = yaml.full_load(input_stream)

# Load config data generated from preprocessing
input_stream = open(os.getcwd() + cfg['PATHS']['INTERPRETABILITY'], 'r')
cfg_gen = yaml.full_load(input_stream)
noncat_features = cfg_gen['NON_CAT_FEATURES']

# Load and partition dataset
df = pd.read_csv(cfg['PATHS']['PROCESSED_OHE_DATA'])
df.drop('ClientID', axis=1, inplace=True)   # Anonymize clients
train_df, test_df = train_test_split(df, test_size=cfg['TRAIN']['TEST_SPLIT'])
train_df, val_df = train_test_split(train_df, test_size=cfg['TRAIN']['VAL_SPLIT'])

# Separate ground truth from dataframe and convert to numpy arrays
train_labels = np.array(train_df.pop('GroundTruth'))
val_labels = np.array(val_df.pop('GroundTruth'))
test_labels = np.array(test_df.pop('GroundTruth'))

# Normalize numerical data
scaler = StandardScaler()
train_df[noncat_features] = scaler.fit_transform(train_df[noncat_features])
val_df[noncat_features] = scaler.transform(val_df[noncat_features])
test_df[noncat_features] = scaler.transform(test_df[noncat_features])

# Convert dataframes to numpy arrays
train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

metrics = [BinaryAccuracy(name="accuracy"), Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]
model = model1(cfg['NN']['MODEL1'], (train_features.shape[-1],), metrics)   # Define model object

history = model.fit(train_features, train_labels, batch_size=cfg['TRAIN']['BATCH_SIZE'], epochs=cfg['TRAIN']['EPOCHS'],
                  validation_data=(val_features, val_labels))

# Run the model on the test set and print the resulting performance metrics.
results = model.evaluate(test_features, test_labels)
print("Results on test set:")
for metric, value in zip(model.metrics_names, results):
  print(metric, ' = ', value)

# Save model weights
save_model(model, cfg['PATHS']['MODEL_WEIGHTS'])


