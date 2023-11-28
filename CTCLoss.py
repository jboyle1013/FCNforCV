import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class CTCLossLayer(Layer):
    def __init__(self, **kwargs):
        super(CTCLossLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # inputs should be in the form [y_pred, labels, input_length, label_length]
        y_pred, labels, input_length, label_length = inputs

        # Compute the CTC loss using Keras backend
        ctc_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
        self.add_loss(ctc_loss)

        # Return the predictions (used for testing, not training)
        return y_pred

    def get_config(self):
        config = super(CTCLossLayer, self).get_config()
        return config