from tensorflow import broadcast_to, shape
from tensorflow.python.keras import backend as K

def f1_loss(recall_factor=1):
    '''
    Custom loss function that calculates F1 score weighted by preference for recall vs. precision (wF1).
    Passing a recall_factor means that you wish to weigh recall {recall_weight} times more than precision in the loss.
    In order to pass a parameter to a custom loss function, the actual loss function must be wrapped within the context
    of a function that receives the custom parameter.
    :param recall_factor: Multiplicative importance of recall with respect to precision.
    :return: A function that computes the f1_loss for ground truth labels and predictions
    '''

    def compute_loss(y_true, y_pred):
        '''
        Computes "recall-weighted" F1 score: wF1 = 2 * (P * R) / (((1-w) * P) + (w * R))
        The coefficient w is calculated from recall_weight.
        :param y_true: Ground truth labels
        :param y_pred: Model's predictions
        :return: The loss, which is calculated from weighted F1 score
        '''

        # Cast predictions and labels, then calculate quantities in numerator and denominator for precision and recall.
        y_pred = K.cast(y_pred, dtype='float32')
        y_true = K.cast(y_true, dtype='float32')
        actual_positives = K.sum(y_true, axis=0)
        predicted_positives = K.sum(y_pred, axis=0)
        true_positives = K.sum(y_true * y_pred, axis=0)

        # Calculate precision and recall
        precision = (true_positives + K.epsilon()) / (predicted_positives + K.epsilon())
        recall = (true_positives + K.epsilon()) / (actual_positives + K.epsilon())

        # Calculate coefficients to be applied in denominator of F1 score to weigh precision and recall accordingly.
        recall_weight = 2 / (recall_factor + 1)
        precision_weight = 2 - recall_weight
        print(precision_weight, recall_weight)

        # Calculate weighted F1 score (wF1)
        weighted_f1 = 2 * (precision * recall) / ((precision_weight * precision) + (recall_weight * recall) +
                                                   K.epsilon())
        loss = 1 - weighted_f1                      # Minimizing this quantity will maximize wF1
        return broadcast_to(loss, shape(y_true))    # Broadcast to shape that takes into account batch size
    return compute_loss