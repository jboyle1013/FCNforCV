import tensorflow as tf
from keras import backend as K

def ctc_loss_function(y_true, y_pred):
    # y_true: tensor (samples, max_string_length) containing the truth labels.
    # y_pred: tensor (samples, time_steps, num_categories) containing the logits.
    # input_length and label_length should be calculated or provided depending on your data.

    # Shape of y_true and y_pred
    label_length = tf.math.count_nonzero(y_true, axis=-1)
    input_length = tf.fill(tf.shape(label_length), tf.shape(y_pred)[1])

    # Compute the CTC loss
    return K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
