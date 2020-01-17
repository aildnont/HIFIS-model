import pandas as pd
import numpy as np
import yaml
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from src.models.models import model1
from src.visualization.visualize import *

def get_class_weights(num_pos, num_neg):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param num_pos: # positive samples
    :param num_neg: # negative samples
    :return: A dictionary containing weights for each class
    '''
    weight_neg = 0.5 * (num_neg + num_pos) / (num_neg)
    weight_pos = 0.5 * (num_neg + num_pos) / (num_pos)
    class_weight = {0: weight_neg, 1: weight_pos}
    print("Class weights: Class 0 = {:.2f}, Class 1 = {:.2f}".format(weight_neg, weight_pos))
    return class_weight

def minority_oversample(X_train, Y_train):
    '''
    Oversample the minority class by duplicating some of its samples
    :param X_train: Training set features
    :param Y_train: Training set labels
    :return: A new training set containing oversampled examples
    '''
    ros = RandomOverSampler(random_state=0)
    X_resampled, Y_resampled = ros.fit_resample(X_train, Y_train)
    print("Train set shape before oversampling: ", X_train.shape, " Train set shape after resampling: ", X_resampled.shape)
    return X_resampled, Y_resampled

def train_model(save_weights=True):
    '''
    Defines and trains HIFIS-v2 model. Prints and logs relevant metrics.
    :param save_weights: A flag indicating whether to save the model weights
    :return: A dictionary of metrics on the test set
    '''

    # Load project config data
    input_stream = open(os.getcwd() + "/config.yml", 'r')
    cfg = yaml.full_load(input_stream)

    # Load config data generated from preprocessing
    input_stream = open(os.getcwd() + cfg['PATHS']['INTERPRETABILITY'], 'r')
    cfg_gen = yaml.full_load(input_stream)
    noncat_features = cfg_gen['NON_CAT_FEATURES']   # Noncategorical features to be scaled
    plot_path = cfg['PATHS']['IMAGES']  # Path for images of matplotlib figures

    # Load and partition dataset
    df = pd.read_csv(cfg['PATHS']['PROCESSED_OHE_DATA'])
    df.drop('ClientID', axis=1, inplace=True)   # Anonymize clients
    num_neg, num_pos = np.bincount(df['GroundTruth'])
    train_split = cfg['TRAIN']['TRAIN_SPLIT']
    val_split = cfg['TRAIN']['VAL_SPLIT']
    test_split = cfg['TRAIN']['TEST_SPLIT']
    train_df, test_df = train_test_split(df, test_size=test_split)
    relative_val_split = val_split / (train_split + val_split)  # Calculate fraction of train_df to be used for validation
    train_df, val_df = train_test_split(train_df, test_size=relative_val_split)

    # Separate ground truth from dataframe and convert to numpy arrays
    Y_train = np.array(train_df.pop('GroundTruth'))
    Y_val = np.array(val_df.pop('GroundTruth'))
    Y_test = np.array(test_df.pop('GroundTruth'))

    # Normalize numerical data
    scaler = StandardScaler()
    train_df[noncat_features] = scaler.fit_transform(train_df[noncat_features])
    val_df[noncat_features] = scaler.transform(val_df[noncat_features])
    test_df[noncat_features] = scaler.transform(test_df[noncat_features])

    # Convert dataframes to numpy arrays
    X_train = np.array(train_df)
    X_val = np.array(val_df)
    X_test = np.array(test_df)

    # Define metrics.
    metrics = [BinaryAccuracy(name="accuracy"), Precision(name="precision"), Recall(name="recall"), AUC(name="auc")]

    # Set callbacks.
    log_dir = cfg['PATHS']['LOGS'] + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
    early_stopping = EarlyStopping(monitor='val_auc', verbose=1, patience=15, mode='max', restore_best_weights=True)
    callbacks = [early_stopping, tensorboard]

    # Define the model.
    model = model1(cfg['NN']['MODEL1'], (X_train.shape[-1],), metrics)   # Build model graph

    # Calculate class weights
    class_weight = None
    if cfg['TRAIN']['CLASS_WEIGHT']:
        class_weight = get_class_weights(num_pos, num_neg)

    # Oversample minority class
    if cfg['TRAIN']['MINORITY_OVERSAMPLE']:
        X_train, Y_train = minority_oversample(X_train, Y_train)

    # Train the model.
    history = model.fit(X_train, Y_train, batch_size=cfg['TRAIN']['BATCH_SIZE'], epochs=cfg['TRAIN']['EPOCHS'],
                      validation_data=(X_val, Y_val), callbacks=callbacks, class_weight=class_weight)

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate(X_test, Y_test)
    test_metrics = {}
    print("Results on test set:")
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[metric] = value
        print(metric, ' = ', value)

    # Visualize metrics about the training process
    test_predictions = model.predict(X_test, batch_size=cfg['TRAIN']['BATCH_SIZE'])
    metrics_to_plot = ['loss', 'auc', 'precision', 'recall']
    plot_metrics(history, metrics_to_plot, file_path=plot_path)
    plot_roc("Test set", Y_test, test_predictions, file_path=plot_path)
    plot_confusion_matrix(Y_test, test_predictions, file_path=plot_path)

    if save_weights:
        save_model(model, cfg['PATHS']['MODEL_WEIGHTS'])        # Save model weights
    return test_metrics


if __name__ == '__main__':
    train_model(save_weights=True)

