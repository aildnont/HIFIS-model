from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2

def model1(config, input_dim, metrics):
    '''
    Defines a Keras model
    :param config: A dictionary of parameters associated with the model architecture
    :param input_dim: The shape of the model input
    :return: a Keras model object with the architecture defined in this method
    '''
    l2_lambda = config['L2_LAMBDA']
    print(l2_lambda, input_dim)
    dropout = config['DROPOUT']

    # Define model architecture.
    model = Sequential()
    model.add(Dense(config['NODES']['DENSE0'], input_shape=input_dim, activation='relu', kernel_regularizer=l2(l2_lambda),
              bias_regularizer=l2(l2_lambda), name="dense0"))
    model.add(Dropout(dropout))
    model.add(Dense(config['NODES']['DENSE1'], activation='relu', kernel_regularizer=l2(l2_lambda),
              bias_regularizer=l2(l2_lambda), name="dense1"))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    # Set model loss function, optimizer, metrics.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    # Print summary of model and return
    model.summary()
    return model