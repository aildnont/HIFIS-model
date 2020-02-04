from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, SGD

def model1(config, input_dim, metrics, hparams=None):
    '''
    Defines a Keras model
    :param config: A dictionary of parameters associated with the model architecture
    :param input_dim: The shape of the model input
    :return: a Keras model object with the architecture defined in this method
    '''

    # Set hyperparameters
    if hparams is None:
        nodes_dense0 = config['NODES']['DENSE0']
        nodes_dense1 = config['NODES']['DENSE1']
        layers = config['LAYERS']
        dropout = config['DROPOUT']
        l2_lambda = config['L2_LAMBDA']
        lr = config['LR']
        optimizer = Adam(learning_rate=lr)
    else:
        nodes_dense0 = hparams['NODES']
        nodes_dense1 = hparams['NODES']
        layers = hparams['LAYERS']
        dropout = hparams['DROPOUT']
        lr = 10 ** hparams['LR']    # Random sampling on logarithmic scale
        beta_1 = 1 - 10 ** hparams['BETA_1']
        beta_2 = 1 - 10 ** hparams['BETA_2']
        l2_lambda = 10 ** hparams['L2_LAMBDA']
        if hparams['OPTIMIZER'] == 'adam':
            optimizer = Adam(learning_rate=lr, beta_1=beta_1, beta_2=beta_2)
        elif hparams['OPTIMIZER'] == 'sgd':
            optimizer = SGD(learning_rate=lr)

    # Define model architecture.
    model = Sequential(name='HIFIS-v2-1')
    model.add(Dense(nodes_dense0, input_shape=input_dim, activation='relu', kernel_regularizer=l2(l2_lambda),
              bias_regularizer=l2(l2_lambda), name="dense0"))
    model.add(Dropout(dropout, name='dropout0'))
    for i in range(1, layers):
        model.add(Dense(nodes_dense1, activation='relu', kernel_regularizer=l2(l2_lambda),
                  bias_regularizer=l2(l2_lambda), name='dense%d'%i))
        model.add(Dropout(dropout, name='dropout%d'%i))
    model.add(Dense(1, activation='sigmoid', name="output"))

    # Set model loss function, optimizer, metrics.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    # Print summary of model and return model object
    if hparams is None:
        model.summary()
    return model