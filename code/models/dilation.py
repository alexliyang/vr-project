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
from initializations.initializations import bilinear_init,identity_init
from keras.utils.data_utils import get_file
from keras.layers import BatchNormalization
dim_ordering = K.image_dim_ordering()

# Paper: https://arxiv.org/pdf/1511.07122.pdf
# Original caffe code: https://github.com/fyu/dilation

TH_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/' \
                         'v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/' \
                         'v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'


def build_dilation(img_shape=(3, None, None), nclasses=11, l2_reg=0.,
               init='glorot_uniform', path_weights=None, load_pretrained=False,
               freeze_layers_from=None,vgg_weights=True):

    # Build network
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    # CONTRACTING PATH

    # Input layer
    inputs = Input(img_shape)
    #padded = ZeroPadding2D(padding=(100, 100), name='pad100')(inputs)

    #Block1
    conv1_1 = Convolution2D(64, 3, 3, init, 'relu', border_mode='same',
                            name='block1_conv1', W_regularizer=l2(l2_reg))(inputs)
    conv1_2 = Convolution2D(64, 3, 3, init, 'relu', border_mode='same',
                      name='block1_conv2', W_regularizer=l2(l2_reg))(conv1_1)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2),name='block1_pool')(conv1_2)

    # Block2
    conv2_1 = Convolution2D(128, 3, 3, init, 'relu', border_mode='same',
                            name='block2_conv1', W_regularizer=l2(l2_reg))(pool1)
    conv2_2 = Convolution2D(128, 3, 3, init, 'relu', border_mode='same',
                            name='block2_conv2', W_regularizer=l2(l2_reg))(conv2_1)
    pool2= MaxPooling2D((2, 2), strides=(2, 2),name='block2_pool')(conv2_2)

    # Block3
    conv3_1 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='block3_conv1', W_regularizer=l2(l2_reg))(pool2)
    conv3_2 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='block3_conv2', W_regularizer=l2(l2_reg))(conv3_1)
    conv3_3 = Convolution2D(256, 3, 3, init, 'relu', border_mode='same',
                            name='block3_conv3', W_regularizer=l2(l2_reg))(conv3_2)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2),name='block3_pool')(conv3_3)

    # Block4
    conv4_1 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='block4_conv1', W_regularizer=l2(l2_reg))(pool3)
    conv4_2 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='block4_conv2', W_regularizer=l2(l2_reg))(conv4_1)
    conv4_3 = Convolution2D(512, 3, 3, init, 'relu', border_mode='same',
                            name='block4_conv3', W_regularizer=l2(l2_reg))(conv4_2)

    vgg_base_model = Model(input=inputs, output=conv4_3)

    vgg_base_in=vgg_base_model.output
    #Block5
    conv5_bn = BatchNormalization(axis=bn_axis, name='block5_bn')(vgg_base_in)

    conv5_1 = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), name='atrous_conv_5_1',
                                  border_mode='same', dim_ordering=dim_ordering, init=identity_init)(conv5_bn)

    conv5_1_relu = Activation('relu')(conv5_1)
    conv5_bn_2 = BatchNormalization(axis=bn_axis, name='block5_bn2')(conv5_1_relu)

    conv5_2 = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), name='atrous_conv_5_2',  border_mode='same',
                                  dim_ordering=dim_ordering, init=identity_init)(conv5_bn_2)

    conv5_2_relu = Activation('relu')(conv5_2)
    conv5_bn3 = BatchNormalization(axis=bn_axis, name='block5_bn3')(conv5_2_relu)

    conv5_3= AtrousConvolution2D(512, 3, 3, atrous_rate=(2,2), name='atrous_conv_5_3',  border_mode='same',
                                 dim_ordering=dim_ordering, init=identity_init)(conv5_bn3)

    conv5_3_relu = Activation('relu')(conv5_3)

    #Block6
    conv6_bn = BatchNormalization(axis=bn_axis, name='block6_bn')(conv5_3)

    conv6= AtrousConvolution2D(1024, 7, 7, atrous_rate=(4, 4), name='atrous_conv_6',
                              border_mode='same', dim_ordering=dim_ordering, init=identity_init)(conv6_bn)

    conv6_relu = Activation('relu')(conv6)
    conv6_relu = Dropout(0.5)(conv6_relu)

    # Block7
    conv7_bn = BatchNormalization(axis=bn_axis, name='block7_bn')(conv6_relu)

    conv7 = AtrousConvolution2D(4096, 1, 1, atrous_rate=(1, 1), name='atrous_conv_7',
                                border_mode='same', dim_ordering=dim_ordering, init=identity_init)(conv7_bn)

    conv7_relu = Activation('relu')(conv7)
    conv7_relu= Dropout(0.5)(conv7_relu)


    #Final block
    convf_bn = BatchNormalization(axis=bn_axis, name='block5_bn')(conv7_relu)

    x = AtrousConvolution2D(nclasses, 1, 1, atrous_rate=(1, 1), name='final_block',
                            border_mode='same', dim_ordering=dim_ordering, init=identity_init)(convf_bn)

    # Appending context block
    upsampling=8
    context_out= context_block(x,[1,1,2,4,8,16,1],nclasses,init=identity_init)
    deconv_out = Deconvolution2D(nclasses, upsampling, upsampling, init=bilinear_init, subsample=(upsampling, upsampling),
                             input_shape=context_out._keras_shape)(context_out)
    # Softmax
    prob = NdSoftmax()(deconv_out)

    # Complete model
    model = Model(input=vgg_base_model.input, output=prob)

    # Load pretrained weights VGG part of the model
    if vgg_weights==True:
        if K.image_dim_ordering() == 'th':
            weights_path = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5',
                                        TH_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')
        else:

             weights_path = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                        TF_WEIGHTS_PATH_NO_TOP,
                                        cache_subdir='models')

        model.load_weights(weights_path,by_name=True)
        print('Loaded pre-trained VGG weights')

    if path_weights:
        print('Loaded pre-trained weights for the full network')
        model.load_weights(weights_path,by_name=True)
      #  load_matcovnet(model, path_weights, n_classes=nclasses)

    # Freeze some layers
    if freeze_layers_from is not None:
       freeze_layers(model, freeze_layers_from)

    return model


def context_block (x, dilation_array,num_classes,init):
    i=0
    for dil in dilation_array:
      x = AtrousConvolution2D(num_classes, 3, 3, atrous_rate=(dil, dil), name='cb_3_{}'.format(i),
                              border_mode='same', dim_ordering=dim_ordering, init=init) (x)

      x = Activation('relu')(x)
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
        freeze_layers_from = 13

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
