# Keras imports
from keras import backend as K
#from keras.layers import Input, merge
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                        ZeroPadding2D)
from keras.layers import Input, BatchNormalization, Activation

#from keras.layers.core import Dropout
from keras.models import Model
from keras.regularizers import l2
#from layers.deconv import Deconvolution2D
from layers.ourlayers import DePool2D, NdSoftmax

from keras.utils.visualize_util import plot

dim_ordering = K.image_dim_ordering()



def build_segnet(img_shape=(3, None, None), nclasses=8, weight_decay=0.,
               freeze_layers_from=None, path_weights=None, basic=False):

    # Regularization warning
    if weight_decay > 0.:
        print ("Regularizing the weights: " + str(weight_decay))


    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # Build network


    # CONTRACTING PATH

    # Input layer
    input_tensor = Input(img_shape)
    #padded = ZeroPadding2D(padding=(100, 100), name='pad100')(inputs)

    x = conv_block(input_tensor, 64, 3, weight_decay, bn_axis, block='1', num='1')
    x = conv_block(x, 64, 3, weight_decay, bn_axis, block='1', num='2')
    pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block1_pool1')(x)

    x = conv_block(pool1, 128, 3, weight_decay, bn_axis, block='2', num='1')
    x = conv_block(x, 128, 3, weight_decay, bn_axis, block='2', num='2')
    pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block2_pool1')(x)

    x = conv_block(pool2, 256, 3, weight_decay, bn_axis, block='3', num='1')
    x = conv_block(x, 256, 3, weight_decay, bn_axis, block='3', num='2')
    x = conv_block(x, 256, 3, weight_decay, bn_axis, block='3', num='3')
    pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block3_pool1')(x)

    x = conv_block(pool3, 512, 3, weight_decay, bn_axis, block='4', num='1')
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='4', num='2')
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='4', num='3')
    pool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block4_pool1')(x)

    x = conv_block(pool4, 512, 3, weight_decay, bn_axis, block='5', num='1')
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='5', num='2')
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='5', num='3')
    pool5 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block5_pool1')(x)


    # DECONTRACTING PATH

    x = DePool2D(pool5, size=(2,2), name='block6_unpool1')(x)
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='6', num='1')
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='6', num='2')
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='6', num='3')

    x = DePool2D(pool4, size=(2,2), name='block7_unpool1')(x)
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='7', num='1')
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='7', num='2')
    x = conv_block(x, 512, 3, weight_decay, bn_axis, block='7', num='3')

    x = DePool2D(pool3, size=(2,2), name='block8_unpool1')(x)
    x = conv_block(x, 256, 3, weight_decay, bn_axis, block='8', num='1')
    x = conv_block(x, 256, 3, weight_decay, bn_axis, block='8', num='2')
    x = conv_block(x, 256, 3, weight_decay, bn_axis, block='8', num='3')

    x = DePool2D(pool2, size=(2,2), name='block9_unpool1')(x)
    x = conv_block(x, 128, 3, weight_decay, bn_axis, block='9', num='1')
    x = conv_block(x, 128, 3, weight_decay, bn_axis, block='9', num='2')

    x = DePool2D(pool1, size=(2,2), name='block10_unpool1')(x)
    x = conv_block(x, 64, 3, weight_decay, bn_axis, block='10', num='1')
    x = conv_block(x, nclasses, 3, weight_decay, bn_axis, block='10', num='2')

    # Softmax
    softmax_segnet = NdSoftmax()(x)

    # Complete model
    model = Model(input=input_tensor, output=softmax_segnet)

    plot(model, to_file='model_1.png', show_shapes=True)

    # Load pretrained Model
    #if path_weights:
    #    load_matcovnet(model, path_weights, n_classes=nclasses)

    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from)

    return model


def conv_block(input_tensor, n_filters, kernel_size, weight_decay, bn_axis, block, num):

    x = Convolution2D(n_filters, kernel_size, kernel_size, border_mode='same',
                      W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay),
                      name='block{}_conv{}'.format(block,num))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='block{}_bn{}'.format(block,num))(x)
    x = Activation('relu', name='block{}_relu{}'.format(block,num))(x)

    return x


def decoder_block(input_tensor, n_filters, kernel_size, weight_decay, bn_axis, unpool_size, block, nclasses=8):

    x = DePool2D(input_tensor, size=unpool_size, name='block{}_unpool1'.format(block))

    x = Convolution2D(n_filters, kernel_size, kernel_size, border_mode='same',
                      W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay),
                      name='block{}_conv1'.format(block))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='block{}_bn1'.format(block))(x)
    x = Activation('relu')(x)

    if block == '10':
        n_filters=nclasses
        x = Convolution2D(n_filters, kernel_size, kernel_size, border_mode='same',
                          W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay),
                          name='block{}_conv2'.format(block))(x)
        return x

    x = Convolution2D(n_filters, kernel_size, kernel_size, border_mode='same',
                      W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay),
                      name='block{}_conv2'.format(block))(x)
    x = BatchNormalization(axis=bn_axis, name='block{}_bn2'.format(block))(x)
    x = Activation('relu')(x)

    if block == '3' or block == '4' or block == '5':
        x = Convolution2D(n_filters, kernel_size, kernel_size, border_mode='same',
                          W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay),
                          name='block{}_conv1'.format(block))(input_tensor)
        x = BatchNormalization(axis=bn_axis, name='block{}_bn1'.format(block))(x)
        x = Activation('relu')(x)

    return x



def custom_sum(tensors):
    t1, t2 = tensors
    return t1 + t2


def custom_sum_shape(tensors):
    t1, t2 = tensors
    return t1


# Freeze layers for finetunning
def freeze_layers(model, freeze_layers_from):
    # Freeze the VGG part only
    if freeze_layers_from == 'base_model':
        print ('   Freezing base model layers')
        freeze_layers_from = 23

    # Show layers (Debug pruposes)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)
    print ('   Freezing from layer 0 to ' + str(freeze_layers_from))

    # Freeze layers
    for layer in model.layers[:freeze_layers_from]:
        layer.trainable = False
    for layer in model.layers[freeze_layers_from:]:
        layer.trainable = True


# Load weights from matconvnet
def load_matcovnet(model, path_weights, n_classes):
    import scipy.io as sio
    import numpy as np

    print('   Loading pretrained model: ' + path_weights)
    # Depending the model has one name or other
    if 'tvg' in path_weights:
        str_filter = 'f'
        str_bias = 'b'
    else:
        str_filter = '_filter'
        str_bias = '_bias'

    # Open the .mat file in python
    W = sio.loadmat(path_weights)

    # Load the parameter values into the model
    num_params = W.get('params').shape[1]
    for i in range(num_params):
        # Get layer name from the saved model
        name = str(W.get('params')[0][i][0])[3:-2]

        # Get parameter value
        param_value = W.get('params')[0][i][1]

        # Load weights
        if name.endswith(str_filter):
            raw_name = name[:-len(str_filter)]

            # Skip final part
            if n_classes == 21 or ('score' not in raw_name and \
                                               'upsample' not in raw_name and \
                                               'final' not in raw_name and \
                                               'probs' not in raw_name):

                print ('   Initializing weights of layer: ' + raw_name)
                print('    - Weights Loaded (FW x FH x FC x K): ' + str(param_value.shape))

                if dim_ordering == 'th':
                    # TH kernel shape: (depth, input_depth, rows, cols)
                    param_value = param_value.T
                    print('    - Weights Loaded (K x FC x FH x FW): ' + str(param_value.shape))
                else:
                    # TF kernel shape: (rows, cols, input_depth, depth)
                    param_value = param_value.transpose((1, 0, 2, 3))
                    print('    - Weights Loaded (FH x FW x FC x K): ' + str(param_value.shape))

                # Load current model weights
                w = model.get_layer(name=raw_name).get_weights()
                print('    - Weights model: ' + str(w[0].shape))
                if len(w) > 1:
                    print('    - Bias model: ' + str(w[1].shape))

                print('    - Weights Loaded: ' + str(param_value.shape))
                w[0] = param_value
                model.get_layer(name=raw_name).set_weights(w)

        # Load bias terms
        if name.endswith(str_bias):
            raw_name = name[:-len(str_bias)]
            if n_classes == 21 or ('score' not in raw_name and
                                           'upsample' not in raw_name and
                                           'final' not in raw_name and
                                           'probs' not in raw_name):
                print ('   Initializing bias of layer: ' + raw_name)
                param_value = np.squeeze(param_value)
                w = model.get_layer(name=raw_name).get_weights()
                w[1] = param_value
                model.get_layer(name=raw_name).set_weights(w)
    return model


if __name__ == '__main__':
    input_shape = [3, 224, 224]
    print (' > Building')
    model = build_segnet(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
