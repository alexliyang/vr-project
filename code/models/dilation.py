# Keras imports
from keras import backend as K
from keras.layers import Input, merge,Activation
from keras.layers.convolutional import (Convolution2D, MaxPooling2D,
                                          ZeroPadding2D,Conv2D, AtrousConvolution2D)
from keras.layers.core import Dropout
from keras.models import Model
from keras.regularizers import l2
#from keras.initializations import Identity
from layers.deconv import Deconvolution2D
from layers.ourlayers import (CropLayer2D, NdSoftmax)
from initializations.initializations import bilinear_init
dim_ordering = K.image_dim_ordering()


# Paper: https://arxiv.org/pdf/1511.07122.pdf
# Original caffe code: https://github.com/fyu/dilation


def build_dilation(img_shape=(3, None, None), nclasses=8, upsampling=8, l2_reg=0.,
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
    #TODO: initialization need sto be identity. Does not work.
    #x = Conv2D(512, 3, strides=(1, 1), padding='same',data_format=dim_ordering, dilation_rate=2, activation='None', use_bias=False,
     #          kernel_initializer=Identity(gain=1.0))(conv4_3)
    conv5_1 = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), name='atrous_conv_5_1',
                                  border_mode='same', dim_ordering=dim_ordering, init=init)(conv4_3)

    conv5_1_relu = Activation('relu')(conv5_1)
    #x = Conv2D(512, 3, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=2, activation='None', use_bias=False,
    #           kernel_initializer=Identity(gain=1.0))(x)
    conv5_2 = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), name='atrous_conv_5_2',  border_mode='same',
                                  dim_ordering=dim_ordering, init=init)(conv5_1_relu)

    conv5_2_relu = Activation('relu')(conv5_2)
   # x = Conv2D(512, 3, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=2, activation='None', use_bias=False,
    #           kernel_initializer=Identity(gain=1.0))(x)
    conv5_3= AtrousConvolution2D(512, 3, 3, atrous_rate=(2,2), name='atrous_conv_5_3',  border_mode='same',
                                 dim_ordering=dim_ordering, init=init)(conv5_2_relu)

    conv5_3_relu = Activation('relu')(conv5_3)

    #Block6
   # x = Conv2D(4096, 7, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=4, activation='None', use_bias=False,
    #           kernel_initializer='identity')(x)
    conv6= AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), name='atrous_conv_6',
                               border_mode='same', dim_ordering=dim_ordering, init=init)(conv5_3_relu)

    conv6_relu = Activation('relu')(conv6)
    conv6_relu = Dropout(0.5)(conv6_relu)

    # Block7
   #x = Conv2D(4096, 1, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=1, activation='None', use_bias=False,
     #          kernel_initializer='identity')(x)
    conv7 = AtrousConvolution2D(4096, 1, 1, atrous_rate=(1, 1), name='atrous_conv_7',
                                border_mode='same', dim_ordering=dim_ordering, init=init)(conv6_relu)

    conv7_relu = Activation('relu')(conv7)
    conv7_relu= Dropout(0.5)(conv7_relu)

    #Final block
    #x = Conv2D(19, 1, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=1, activation='None', use_bias=False,
     #          kernel_initializer='identity')(x)

    x = AtrousConvolution2D(19, 1, 1, atrous_rate=(1, 1), name='final_block',
                            border_mode='same', dim_ordering=dim_ordering, init=init)(conv7_relu)

    # Appending context block

    context_out= context_block(x,[1,1,2,4,8,16,32,64,1],19,init)
    deconv_out = Deconvolution2D(19, upsampling, upsampling, init=bilinear_init, subsample=(upsampling, upsampling),
                             input_shape=context_out._keras_shape)(context_out)

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


def context_block (x, dilation_array,num_classes,init):
    i=0
    for dil in dilation_array:
      # x=Conv2D(num_classes, 3, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=dil,  activation='None', use_bias=False,kernel_initializer='identity')(x)
      x = AtrousConvolution2D(num_classes, 3, 3, atrous_rate=(dil, dil), name='cb_3_{}'.format(i),
                              border_mode='same', dim_ordering=dim_ordering, init=init) (x)

      x = Activation('relu')(x)
    #x = Conv2D(num_classes, 1, strides=(1, 1), padding='same', data_format=dim_ordering, dilation_rate=1,
     #          kernel_initializer='identity')(x)
    i = i + 1
    x = AtrousConvolution2D(num_classes, 1, 1, atrous_rate=(1, 1),name='cb_final_conv',
                               border_mode='same', dim_ordering=dim_ordering, init=init)(x)

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
