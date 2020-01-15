import pandas as pd
import numpy as np
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.models import save_model
from src.models.models import model1


# Load project config data
input_stream = open(os.getcwd() + "/config.yml", 'r')
config = yaml.full_load(input_stream)

# Load and partition dataset
df = pd.read_csv(config['PATHS']['PROCESSED_OHE_DATA'])
df.replace('Unknown', 0, inplace=True)
train_df, test_df = train_test_split(df, test_size=config['TRAIN']['TEST_SPLIT'])
train_df, val_df = train_test_split(train_df, test_size=config['TRAIN']['VAL_SPLIT'])

# Separate ground truth from dataframe and convert datasets to numpy arrays
train_labels = np.array(train_df.pop('GroundTruth'))
val_labels = np.array(val_df.pop('GroundTruth'))
test_labels = np.array(test_df.pop('GroundTruth'))
train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

# Normalize data
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

metrics = [Precision(), Recall(), BinaryAccuracy()]
model = model1(config['NN']['MODEL1'], (train_features.shape[-1],), metrics)

history = model.fit(train_features, train_labels, batch_size=config['TRAIN']['BATCH_SIZE'], epochs=config['TRAIN']['EPOCHS'],
                  validation_data=(val_features, val_labels))
results = model.evaluate(test_features, test_labels)

print("Results on test set:")
for result in results:
    print(result)

# Save model weights
save_model(model, config['PATHS']['MODEL_WEIGHTS'])


