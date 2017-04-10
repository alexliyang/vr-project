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
from layers.ourlayers import DePool2D, NdSoftmax, CropLayer2D

from keras.utils.visualize_util import plot

dim_ordering = K.image_dim_ordering()



def build_segnet(img_shape=(None, None, 3), nclasses=8, weight_decay=0.,
               freeze_layers_from=None, path_weights=None, basic=False):

    # Regularization warning
    if weight_decay > 0.:
        print ("Regularizing the weights: " + str(weight_decay))


    # Set axis in which to do batch normalization
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1


    # Build network

    # Input layer
    input_tensor = Input(img_shape)

    # Pad image to avoid size problems with pooling-unpooling
    padded = ZeroPadding2D(padding=(100, 100), name='pad100')(input_tensor)

    if not basic:

        # CONTRACTING PATH
        x = conv_block(padded, 64, 3, weight_decay, bn_axis, block='1', num='1')
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

        x = DePool2D(pool2d_layer=pool5, size=(2,2), name='block6_unpool1')(pool5)
        x = conv_block(x, 512, 3, weight_decay, bn_axis, block='6', num='1')
        x = conv_block(x, 512, 3, weight_decay, bn_axis, block='6', num='2')
        x = conv_block(x, 512, 3, weight_decay, bn_axis, block='6', num='3')

        x = DePool2D(pool2d_layer=pool4, size=(2,2), name='block7_unpool1')(x)
        x = conv_block(x, 512, 3, weight_decay, bn_axis, block='7', num='1')
        x = conv_block(x, 512, 3, weight_decay, bn_axis, block='7', num='2')
        x = conv_block(x, 512, 3, weight_decay, bn_axis, block='7', num='3')

        x = DePool2D(pool2d_layer=pool3, size=(2,2), name='block8_unpool1')(x)
        x = conv_block(x, 256, 3, weight_decay, bn_axis, block='8', num='1')
        x = conv_block(x, 256, 3, weight_decay, bn_axis, block='8', num='2')
        x = conv_block(x, 256, 3, weight_decay, bn_axis, block='8', num='3')

        x = DePool2D(pool2d_layer=pool2, size=(2,2), name='block9_unpool1')(x)
        x = conv_block(x, 128, 3, weight_decay, bn_axis, block='9', num='1')
        x = conv_block(x, 128, 3, weight_decay, bn_axis, block='9', num='2')

        x = DePool2D(pool2d_layer=pool1, size=(2,2), name='block10_unpool1')(x)
        x = conv_block(x, 64, 3, weight_decay, bn_axis, block='10', num='1')
        x = conv_block(x, nclasses, 3, weight_decay, bn_axis, block='10', num='2')

    elif basic:

        # CONTRACTING PATH
        x = conv_block(padded, 64, 7, weight_decay, bn_axis, block='1', num='1')
        x = conv_block(x, 64, 7, weight_decay, bn_axis, block='1', num='2')
        pool1 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block1_pool1')(x)

        x = conv_block(pool1, 128, 7, weight_decay, bn_axis, block='2', num='1')
        x = conv_block(x, 128, 7, weight_decay, bn_axis, block='2', num='2')
        pool2 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block2_pool1')(x)

        x = conv_block(pool2, 256, 7, weight_decay, bn_axis, block='3', num='1')
        x = conv_block(x, 256, 7, weight_decay, bn_axis, block='3', num='2')
        x = conv_block(x, 256, 7, weight_decay, bn_axis, block='3', num='3')
        pool3 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block3_pool1')(x)

        x = conv_block(pool3, 512, 7, weight_decay, bn_axis, block='4', num='1')
        x = conv_block(x, 512, 7, weight_decay, bn_axis, block='4', num='2')
        x = conv_block(x, 512, 7, weight_decay, bn_axis, block='4', num='3')
        pool4 = MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block4_pool1')(x)


        # DECONTRACTING PATH

        x = DePool2D(pool2d_layer=pool4, size=(2,2), name='block7_unpool1')(pool4)
        x = conv_block(x, 512, 7, weight_decay, bn_axis, block='7', num='1', deconv_basic=True)
        x = conv_block(x, 512, 7, weight_decay, bn_axis, block='7', num='2', deconv_basic=True)
        x = conv_block(x, 512, 7, weight_decay, bn_axis, block='7', num='3', deconv_basic=True)

        x = DePool2D(pool2d_layer=pool3, size=(2,2), name='block8_unpool1')(x)
        x = conv_block(x, 256, 7, weight_decay, bn_axis, block='8', num='1', deconv_basic=True)
        x = conv_block(x, 256, 7, weight_decay, bn_axis, block='8', num='2', deconv_basic=True)
        x = conv_block(x, 256, 7, weight_decay, bn_axis, block='8', num='3', deconv_basic=True)

        x = DePool2D(pool2d_layer=pool2, size=(2,2), name='block9_unpool1')(x)
        x = conv_block(x, 128, 7, weight_decay, bn_axis, block='9', num='1', deconv_basic=True)
        x = conv_block(x, 128, 7, weight_decay, bn_axis, block='9', num='2', deconv_basic=True)

        x = DePool2D(pool2d_layer=pool1, size=(2,2), name='block10_unpool1')(x)
        x = conv_block(x, 64, 7, weight_decay, bn_axis, block='10', num='1', deconv_basic=True)
        x = conv_block(x, nclasses, 7, weight_decay, bn_axis, block='10', num='2', deconv_basic=True)


    # Recover the image's original size
    x = CropLayer2D(input_tensor, name='score')(x)

    # Softmax
    softmax_segnet = NdSoftmax()(x)

    # Complete model
    model = Model(input=input_tensor, output=softmax_segnet)

    #TODO: load weights from caffe
    # Load pretrained Model
    #if path_weights:
    #    load_matcovnet(model, path_weights, n_classes=nclasses)

    #TODO: review freeze layers
    # Freeze some layers
    if freeze_layers_from is not None:
        freeze_layers(model, freeze_layers_from)

    return model


def conv_block(input_tensor, n_filters, kernel_size, weight_decay, bn_axis, block, num, deconv_basic=False):

    x = Convolution2D(n_filters, kernel_size, kernel_size, border_mode='same',
                      W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay), init='he_normal',
                      name='block{}_conv{}'.format(block,num))(input_tensor)
    x = BatchNormalization(axis=bn_axis, name='block{}_bn{}'.format(block,num))(x)
    if not deconv_basic:
        x = Activation('relu', name='block{}_relu{}'.format(block,num))(x)

    return x


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


if __name__ == '__main__':
    input_shape = [3, 224, 224]
    print (' > Building')
    model = build_segnet(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
