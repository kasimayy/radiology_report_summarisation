import keras.backend as K
import tensorflow as tf
import numpy as np

def categorical_accuracy_with_variable_timestep(y_true, y_pred):
    y_true = y_true[:, :-1, :]  # Discard the last timestep
    y_pred = y_pred[:, :-1, :]  # Discard the last timestep

    # Flatten the timestep dimension
    shape = tf.shape(y_true)
    y_true = tf.reshape(y_true, [-1, shape[-1]])
    y_pred = tf.reshape(y_pred, [-1, shape[-1]])

    # Discard rows that are all zeros as they represent padding words.
    is_zero_y_true = tf.equal(y_true, 0)
    is_zero_row_y_true = tf.reduce_all(is_zero_y_true, axis=-1)
    y_true = tf.boolean_mask(y_true, ~is_zero_row_y_true)
    y_pred = tf.boolean_mask(y_pred, ~is_zero_row_y_true)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_true, axis=1),
                                              tf.argmax(y_pred, axis=1)),
                                    dtype=tf.float32))
    return accuracy

# def recall(y_true, y_pred):
#     data0 = tf.zeros(tf.shape(y_true), dtype = tf.float32)
#     data1 = tf.ones(tf.shape(y_true), dtype = tf.float32)

#     y_true_p = y_true*data1
#     y_true_n = data1-y_true
#     y_pred_p = y_pred*data1
#     y_pred_n = data1-y_pred

#     TP = K.sum(y_true_p*y_pred_p, axis=-1)
#     FN = K.sum(y_true_p*y_pred_n, axis=-1)

#     recall = TP / (TP + FN + K.epsilon())
#     return recall

# def specificity(y_true, y_pred):
#     data0 = tf.zeros(tf.shape(y_true), dtype = tf.float32)
#     data1 = tf.ones(tf.shape(y_true), dtype = tf.float32)

#     y_true_p = y_true*data1
#     y_true_n = data1-y_true
#     y_pred_p = y_pred*data1
#     y_pred_n = data1-y_pred

#     TN = K.sum(y_true_n*y_pred_n, axis=-1)
#     FP = K.sum(y_true_n*y_pred_p, axis=-1)

#     specificity = TN / (TN + FP + K.epsilon())
#     return specificity

def precision(y_true, y_pred):
    """Precision metric.
    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.
    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_np(y_true, y_pred):
    y_pred_ = np.array([y_pred > 0.5])*1.0
    y_pred = y_pred_[0]
    
    TP = np.sum(y_true*y_pred)
    pred_P = np.sum(y_pred)
    total_precision = TP/(pred_P + 1e-7)
    
    TP = np.sum(y_true*y_pred, axis=0)
    pred_P = np.sum(y_pred, axis=0)
    precision = TP/(pred_P + 1e-7)
    avg_over_classes = np.mean(precision)
    
    TP = np.sum(y_true*y_pred, axis=1)
    pred_P = np.sum(y_pred, axis=1)
    precision = TP/(pred_P + 1e-7)
    avg_over_samples = np.mean(precision)
    
    return total_precision, avg_over_classes, avg_over_samples

def recall_np(y_true, y_pred):
    y_pred_ = np.array([y_pred > 0.5])*1.0
    y_pred = y_pred_[0]
    
    TP = np.sum(y_true*y_pred)
    all_P = np.sum(y_true)
    total_recall = TP/(all_P + 1e-7)
    
    TP = np.sum(y_true*y_pred, axis=0)
    all_P = np.sum(y_true, axis=0)
    recall = TP/(all_P + 1e-7)
    avg_over_classes = np.mean(recall)
    
    TP = np.sum(y_true*y_pred, axis=1)
    all_P = np.sum(y_true, axis=1)
    recall = TP/(all_P + 1e-7)
    avg_over_samples = np.mean(recall)
    
    return total_recall, avg_over_classes, avg_over_samples

def multilabel_confusion_matrix(y_true, y_pred, sum_per='class'):
    '''Computes multilabel confusion matrix for all samples. 
    
    Parameters
    ----------
    y_true: np.array, shape: (samples, classes)
    y_pred: np.array, shape: (samples, classes)
    sum_over: str, 'class': returns sums of TP,TN,FP,FN per class
    'samples': returns sum of TP,TN,FP,FN per sample
    
    Returns
    -------
    confusion_matrix: [[TP, FP], [FN, TN]] per class or per sample
    '''
    
    assert sum_per in ['class', 'sample'], 'sum_per option must be either "class" or "sample"'
    axis = 0 if sum_per=='class' else 1
    
    data1 = np.ones(y_true.shape[1])
    
    y_true_n = data1-y_true
    y_pred_n = data1-y_pred
    
    TP = np.sum(y_true*y_pred, axis=axis).astype(np.float32)
    TN = np.sum(y_true_n*y_pred_n, axis=axis).astype(np.float32)
    FP = np.sum(y_true_n*y_pred, axis=axis).astype(np.float32)
    FN = np.sum(y_true*y_pred_n, axis=axis).astype(np.float32)
    
    TP = TP.reshape(len(TP),1)
    TN = TN.reshape(len(TN),1)
    FP = FP.reshape(len(FP),1)
    FN = FN.reshape(len(FN),1)
    
    conf_mat = np.concatenate((TP,FP,FN,TN), axis=1)
    split_mats = np.vsplit(conf_mat, conf_mat.shape[0])
    confusion_matrix = []
    for mat in split_mats:
        confusion_matrix.append(mat.reshape(2,2))

    return confusion_matrix

def binary_accuracy(y_true, y_pred):
     return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def binary_accuracy_np(y_true, y_pred):
    return np.mean(np.mean(np.equal(y_true, np.round(y_pred)), axis=-1))

    
    
    
    
    
    
    
    
    
    
    