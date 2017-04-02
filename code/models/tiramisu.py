from __future__ import print_function

from keras import backend as K
from keras.layers import Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from keras.layers.core import Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.layers import Activation, BatchNormalization, MaxPooling2D
from code.layers.ourlayers import NdSoftmax

from code.models.densenetFCN import denseblock

# Batch normalization dimensions
dim_ordering = K.image_dim_ordering()
if dim_ordering == "th":
    bn_axis = 1
else:
    bn_axis = -1


""" MODEL BUILDERS """
"""
Paper: https://arxiv.org/abs/1611.09326
Implementation based on the following Theano / Lasagne code: https://github.com/SimJeg/FC-DenseNet
"""


def build_tiramisu_fc67(img_shape=(None, None, 3), nclasses=8, weight_decay=1e-4, compression=0, dropout=0.2,
                        freeze_layers_from=None,
                        nb_filter=48):
    # Parameters of the network
    n_dense_blocks = 5  # This does not include the "transition" dense block between the down and upsampling paths
    n_layers_block = 5  # Dense layers per dense block
    growth_rate = 16  # Growth rate of dense blocks, k in DenseNet paper
    compression = 1 - compression  # Compression factor applied in Transition Down (only in case of OOM problems)

    tiramisu_model, network = tiramisu_network(
        img_shape, n_dense_blocks, n_layers_block, growth_rate, nclasses, weight_decay, compression, dropout,
        'Tiramisu_FC67', freeze_layers_from, nb_filter
    )

    return tiramisu_model


def build_tiramisu_fc103(img_shape=(None, None, 3), nclasses=8, weight_decay=1e-4, compression=0, dropout=0.2,
                         freeze_layers_from=None,
                         nb_filter=48):
    # TODO: Parameters of the network
    n_dense_blocks = 5  # This does not include the "transition" dense block between the down and upsampling paths
    n_layers_block = 5  # Dense layers per dense block
    growth_rate = 16  # Growth rate of dense blocks, k in DenseNet paper
    compression = 1 - compression  # Compression factor applied in Transition Down (only in case of OOM problems)

    tiramisu_model, network = tiramisu_network(
        img_shape, n_dense_blocks, n_layers_block, growth_rate, nclasses, weight_decay, compression, dropout,
        'Tiramisu_FC103', freeze_layers_from, nb_filter
    )

    return tiramisu_model


""" BUILDING BLOCKS """


def tiramisu_network(img_shape, n_dense_blocks, n_layers_block, growth_rate,
                     nclasses, weight_decay, compression, dropout, network_name,
                     freeze_layers_from=None, nb_filter=48):
    # Placeholder for the feature maps in each dense block, transition down or transition up
    net = dict()

    # Initial convolution
    net['input'] = Input(shape=img_shape)
    x = Convolution2D(nb_filter, 3, 3,
                      init="he_uniform",
                      border_mode="same",
                      name="initial_conv2D",
                      W_regularizer=l2(weight_decay))(net['input'])
    net['init_conv'] = x

    # Dense blocks + Transition down in the downsampling path
    for block_idx in range(n_dense_blocks):
        # Dense block
        x, nb_filter = denseblock(x, n_layers_block, nb_filter, growth_rate,
                                  dropout_rate=dropout,
                                  weight_decay=weight_decay)
        feature_name = 'db_{}'.format(block_idx + 1)
        net[feature_name] = x

        # Compression
        nb_filter = int(compression * nb_filter)

        # Transition Down
        x = transition_tiramisu(x, nb_filter, dropout_rate=dropout, weight_decay=weight_decay)
        feature_name = 'td_{}'.format(block_idx + 1)
        net[feature_name] = x

    # The last denseblock does not have a transition
    x, nb_filter = denseblock(x, n_layers_block, nb_filter, growth_rate,
                              dropout_rate=dropout,
                              weight_decay=weight_decay)
    feature_name = 'db_{}'.format(n_dense_blocks + 1)
    net[feature_name] = x

    # TODO: Upsampling path

    # TODO: Bottleeck

    tiramisu_model = Model(input=[net['input']], output=[x], name=network_name)

    # Freeze some layers
    if freeze_layers_from is not None:
        if freeze_layers_from == 'base_model':
            raise ValueError('Freezing the base_model is not supported for network {}'.format(network_name))
        else:
            for i, layer in enumerate(model.layers):
                print(i, layer.name)
            print('   Freezing from layer 0 to ' + str(freeze_layers_from))
            for layer in model.layers[:freeze_layers_from]:
                layer.trainable = False
            for layer in model.layers[freeze_layers_from:]:
                layer.trainable = True

    return tiramisu_model, net


def transition_tiramisu(x, nb_filter, dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 1x1Conv2D, optional dropout and Maxpooling2D
    :param x: keras model
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: model
    :rtype: keras model, after applying batch_norm, relu-conv, dropout, maxpool
    """
    if K.image_dim_ordering() == "th":
        bn_axis = 1
    else:
        bn_axis = -1

    x = BatchNormalization(mode=0,
                           axis=bn_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Convolution2D(nb_filter, 1, 1,
                      init="he_uniform",
                      border_mode="same",
                      bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    return x


if __name__ == '__main__':
    input_shape = [3, 224, 224]
    print(' > Building Tiramisu FC67')
    model = build_tiramisu_fc67(input_shape, 11, 0.)
    print(' > Compiling Tiramisu FC67')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()

    print(' > Building Tiramisu FC103')
    model = build_tiramisu_fc103(input_shape, 11, 0.)
    print(' > Compiling Tiramisu FC103')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
