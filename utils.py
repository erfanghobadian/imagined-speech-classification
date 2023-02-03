import numpy as np
import tensorflow as tf


def std(x_train, x_test, x_val):
    mean = x_train.mean(axis=0)
    std_val = x_train.std(axis=0)
    x_train_std = (x_train - mean) / std_val
    x_test_std = (x_test - mean) / std_val
    x_val_std = (x_val - mean) / std_val
    return x_train_std, x_test_std, x_val_std


def fft(epoch):
    nfft = epoch.shape[1]
    freq = np.empty(1, dtype=int)
    freq[0] = int(nfft / 2)
    fft_res = np.fft.fft(epoch, n=nfft)
    fft_abs = np.abs(fft_res[:, :freq[0]])
    x = fft_abs * fft_abs
    x[x == 0] = 0.00001
    y = 20 * np.log(x)
    return y


def cov(epoch):
    c = np.cov(epoch)
    return c


def EEGNet(nb_classes, Chans=62, Samples=250,
           dropoutRate=0.5, kernLength=64, F1=8,
           D=2, F2=16, norm_rate=0.25, dropoutType='Dropout'):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = tf.keras.layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = tf.keras.layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = tf.keras.layers.Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = tf.keras.layers.Conv2D(F1, (1, kernLength), padding='same',
                                    input_shape=(Chans, Samples, 1),
                                    use_bias=False)(input1)
    block1 = tf.keras.layers.BatchNormalization()(block1)
    block1 = tf.keras.layers.DepthwiseConv2D((Chans, 1), use_bias=False,
                                             depth_multiplier=D,
                                             depthwise_constraint=tf.keras.constraints.max_norm(1.))(block1)
    block1 = tf.keras.layers.BatchNormalization()(block1)
    block1 = tf.keras.layers.Activation('elu')(block1)
    block1 = tf.keras.layers.AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = tf.keras.layers.SeparableConv2D(F2, (1, 16),
                                             use_bias=False, padding='same')(block1)
    block2 = tf.keras.layers.BatchNormalization()(block2)
    block2 = tf.keras.layers.Activation('elu')(block2)
    block2 = tf.keras.layers.AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = tf.keras.layers.Flatten(name='flatten')(block2)

    dense = tf.keras.layers.Dense(nb_classes, name='dense',
                                  kernel_constraint=tf.keras.constraints.max_norm(norm_rate))(flatten)
    softmax = tf.keras.layers.Activation('softmax', name='softmax')(dense)

    return tf.keras.models.Model(inputs=input1, outputs=softmax)