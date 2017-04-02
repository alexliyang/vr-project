# Keras imports
from keras import backend as K
from keras.layers import Input, merge,Activation
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                          ZeroPadding2D,Conv2D)
from keras.layers.core import Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.initializations import Identity
from layers.deconv import Deconvolution2D
from layers.ourlayers import (CropLayer2D, NdSoftmax)

dim_ordering = K.image_dim_ordering()


# Paper: https://arxiv.org/pdf/1511.07122.pdf
# Original caffe code: https://github.com/fyu/dilation


def build_dilation(img_shape=(3, None, None), nclasses=8, l2_reg=0.,
               init='glorot_uniform', path_weights=None,
               freeze_layers_from=None):
    # Regularization warning
    if l2_reg > 0.:
        print ("Regularizing the weights: " + str(l2_reg))

    # Build network

    # CONTRACTING PATH

    # Input layer
    inputs = Input(img_shape)
    #padded = ZeroPadding2D(padding=(100, 100), name='pad100')(inputs)

    #Block1
    conv1_1 = Convolution2D(64, 3, 3, init, 'relu', border_mode='same',
                            name='conv1_1', W_regularizer=l2(l2_reg))(inputs)
    conv1_2 = Convolution2D(64, 3, 3, init, 'relu', border_mode='same',
                      name='conv1_2', W_regularizer=l2(l2_reg))(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    # Block2
    conv2_1 = Convolution2D(128, 3, 3, init, 'relu', border_mode='same',
                            name='conv2_1', W_regularizer=l2(l2_reg))(pool1)
    conv2_2 = Convolution2D(128, 3, 3, init, 'relu', border_mode='same',
                            name='conv2_2', W_regularizer=l2(l2_reg))(conv2_1)
    pool2= MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    # Block3
    conv3_1 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='conv3_1', W_regularizer=l2(l2_reg))(pool2)
    conv3_2 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='conv3_2', W_regularizer=l2(l2_reg))(conv3_1)
    conv3_3 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='conv3_3', W_regularizer=l2(l2_reg))(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    # Block4
    conv4_1 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv4_1', W_regularizer=l2(l2_reg))(pool3)
    conv4_2 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv4_2', W_regularizer=l2(l2_reg))(conv4_1)
    conv4_3 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='conv4_3', W_regularizer=l2(l2_reg))(conv4_2)

    #Block5
    x = Conv2D(512, 3, strides=(1, 1), padding='same',data_format=dim_ordering, dilation_rate=2, activation='None', use_bias=False,
               kernel_initializer=Identity(gain=1.0))(conv4_3)
    x = Activation('relu')(x)
    x = Conv2D(512, 3, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=2, activation='None', use_bias=False,
               kernel_initializer=Identity(gain=1.0))(x)
    x = Activation('relu')(x)
    x = Conv2D(512, 3, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=2, activation='None', use_bias=False,
               kernel_initializer=Identity(gain=1.0))(x)
    x = Activation('relu')(x)

    #Block6
    x = Conv2D(4096, 7, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=4, activation='None', use_bias=False,
               kernel_initializer='identity')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Block7
    x = Conv2D(4096, 1, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=1, activation='None', use_bias=False,
               kernel_initializer='identity')(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    #Final block
    x = Conv2D(19, 1, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=1, activation='None', use_bias=False,
               kernel_initializer='identity')(x)
    # Appending context block

    context_out= context_block(x,[1,1,2,4,8,16,32,64,1],19)

    deconv_out = Deconvolution2D(19, 16, 16, context_out._keras_shape, init,
                             'bilinear', border_mode='valid', subsample=(8, 8),bias=False,
                             name='score2', W_regularizer=l2(l2_reg))(context_out)


    # Softmax
    prob = NdSoftmax()(deconv_out)

    # Complete model
    model = Model(input=inputs, output=prob)

    # Load pretrained Model
   # if path_weights:
      #  load_matcovnet(model, path_weights, n_classes=nclasses)

    # Freeze some layers
    #if freeze_layers_from is not None:
    #    freeze_layers(model, freeze_layers_from)

    return model


def context_block (x, dilation_array,num_classes):
    for dil in range(dilation_array):
       x=Conv2D(num_classes, 3, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=dil,  activation='None', use_bias=False,kernel_initializer='identity')(x)
       x = Activation('relu')(x)
    x = Conv2D(num_classes, 1, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=1,
               kernel_initializer='identity')(x)
    x = Activation('relu')(x)
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
    model = build_dilation(input_shape, 11, 0.)
    print (' > Compiling')
    model.compile(loss="categorical_crossentropy", optimizer="rmsprop")
    model.summary()
