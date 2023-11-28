import tensorflow as tf
from keras import backend as K
from keras.models import Model


tf.config.run_functions_eagerly(True)


def create_ctc_loss_function():

    def ctc_loss_function(args):
        y_pred, labels, input_length, label_length = args
        print("Received arguments:", labels, y_pred, input_length, label_length)
        # Your loss calculation code here
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

    return ctc_loss_function