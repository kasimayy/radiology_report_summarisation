import numpy as np
import keras.backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf


def categorical_crossentropy_from_logits(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                 logits=y_pred)
    return loss

def binary_recall_specificity(y_true, y_pred, recall_weight):
    '''Calculate loss on a batch as a function of recall and specificity.
    
    Parameters
    ----------
    y_true: K.variable, batch of true binary values, shape = (batch_size, features)
    y_pred: K, variable, batch of predicted binary values, shape = (batch_size, features)
    recall_weight: float, 0.0<=w<=1.0
    spec_weight: float, 1.0-recall_weight
    
    Returns
    -------
    loss: float
    '''
    
    data0 = tf.zeros(tf.shape(y_true), dtype = tf.float32)
    data1 = tf.ones(tf.shape(y_true), dtype = tf.float32)

    y_true_p = y_true*data1
    y_true_n = data1-y_true
    y_pred_p = y_pred*data1
    y_pred_n = data1-y_pred

    TP = K.sum(y_true_p*y_pred_p, axis=-1)
    TN = K.sum(y_true_n*y_pred_n, axis=-1)
    FP = K.sum(y_true_n*y_pred_p, axis=-1)
    FN = K.sum(y_true_p*y_pred_n, axis=-1)

    specificity = TN / (TN + FP + K.epsilon())
    recall = TP / (TP + FN + K.epsilon())

    #avg_specificity = K.mean(specificity, axis=-1)
    #avg_recall = K.mean(recall, axis=-1)
    
    return 1.0 - (recall_weight*recall + (1-recall_weight)*specificity)

def binary_recall_specificity_loss(recall_weight):
    '''Wrapper function for binary_recall_specificity
    '''

    def recall_spec_loss(y_true, y_pred):
        return binary_recall_specificity(y_true, y_pred, recall_weight)

    # Returns the (y_true, y_pred) loss function
    return recall_spec_loss

def combined_loss(bce_weight, recall_weight):
    '''Wrapper function for combined loss of binary crossentropy and recall_specificity
    '''
    def loss(y_true, y_pred):
        return bce_weight*binary_crossentropy(y_true, y_pred)+(1-bce_weight)*binary_recall_specificity(y_true, y_pred, recall_weight)
    
    return loss