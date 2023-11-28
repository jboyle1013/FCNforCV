from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Lambda
from keras.layers import Dense, Bidirectional, LSTM, Reshape, Add
from keras.layers import LeakyReLU, DepthwiseConv2D, SeparableConv2D
from keras.models import Model
import keras.backend as K
from CTCLoss import ctc_loss

import tensorflow as tf

tf.config.run_functions_eagerly(True)

def squeeze(x):
    return K.squeeze(x, 1)


def CNN_model(max_string_length):
    char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    dropout_rate = 0.25
    lstm_units = 256

    # Improved model structure with advanced convolutional layers and residual connections
    model_inputs = Input(shape=(32, 128, 1), name='input')
    # Additional inputs for CTC loss

    # First convolutional block with residual connection
    # First convolutional block with residual connection
    conv_1 = Conv2D(64, (3, 3), padding='same')(model_inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = LeakyReLU()(conv_1)
    conv_1_res = Conv2D(64, (3, 3), padding='same')(conv_1)
    conv_1_res = BatchNormalization()(conv_1_res)
    conv_1_res = LeakyReLU()(conv_1_res)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_1_res)

    # Adjusting the shortcut path to match dimensions
    shortcut_1 = Conv2D(64, (1, 1), strides=(2, 2), padding='same')(conv_1)
    res_1 = Add()([shortcut_1, pool_1])

    # Second convolutional block with depthwise separable convolutions
    conv_2 = SeparableConv2D(128, (3, 3), padding='same')(res_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = LeakyReLU()(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=2)(conv_2)

    # Third convolutional block with dilated convolutions
    conv_3 = Conv2D(256, (3, 3), dilation_rate=2, padding='same')(pool_2)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = LeakyReLU()(conv_3)

    # Further layers continue as before
    conv_4 = Conv2D(256, (3, 3), padding='same')(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = LeakyReLU()(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 1))(conv_4)

    conv_5 = Conv2D(512, (3, 3), padding='same')(pool_4)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = LeakyReLU()(conv_5)

    conv_6 = Conv2D(512, (3, 3), padding='same')(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = LeakyReLU()(conv_6)
    pool_6 = MaxPooling2D(pool_size=(2, 1))(conv_6)

    conv_7 = Conv2D(512, (2, 2), activation='relu')(pool_6)

    # Correct reshaping for LSTM layers
    # Compute the new shape: The number of timesteps is the second dimension of the output of conv_7
    # Using the named function in a Lambda layer
    squeezed = Lambda(squeeze)(conv_7)

    # Bidirectional LSTM layers
    blstm_1 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))(squeezed)
    blstm_2 = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))(blstm_1)

    outputs = Dense(len(char_list) + 1, activation='softmax')(blstm_2)

    test_model = Model(model_inputs, outputs)

    labels = Input(name='labels', shape=[max_string_length], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')



    loss_out = Lambda(ctc_loss, output_shape=(1,), name='ctc')([outputs, labels, input_length, label_length])


    model = Model(inputs=[model_inputs, labels, input_length, label_length], outputs=loss_out)

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='sgd', metrics=['accuracy'])


    model.summary()
    print(f'Total number of layers: {len(model.layers)}')

    return model


if __name__ == "__main__":
    CNN_model()
