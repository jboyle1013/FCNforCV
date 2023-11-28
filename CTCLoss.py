import tensorflow as tf
from keras import backend as K
from keras.models import Model

def ctc_loss(args):
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
