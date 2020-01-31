from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def model1(config, input_dim, metrics, hparams=None):
    '''
    Defines a Keras model
    :param config: A dictionary of parameters associated with the model architecture
    :param input_dim: The shape of the model input
    :return: a Keras model object with the architecture defined in this method
    '''
    l2_lambda = config['L2_LAMBDA']
    if hparams is None:
        dropout = config['DROPOUT']
        lr = config['LR']
        nodes_dense0 = config['NODES']['DENSE0']
        nodes_dense1 = config['NODES']['DENSE1']
    else:
        dropout = hparams['DROPOUT']
        lr = hparams['LR']
        nodes_dense0 = hparams['NODES']
        nodes_dense1 = hparams['NODES']

    # Define model architecture.
    model = Sequential(name='HIFIS-v2-1')
    model.add(Dense(nodes_dense0, input_shape=input_dim, activation='relu', kernel_regularizer=l2(l2_lambda),
              bias_regularizer=l2(l2_lambda), name="dense0"))
    model.add(Dropout(dropout, name='dropout0'))
    model.add(Dense(nodes_dense1, activation='relu', kernel_regularizer=l2(l2_lambda),
              bias_regularizer=l2(l2_lambda), name="dense1"))
    model.add(Dropout(dropout, name='dropout1'))
    model.add(Dense(1, activation='sigmoid', name="output"))

    # Set model loss function, optimizer, metrics.
    optimizer = Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    # Print summary of model and return
    model.summary()
    return model