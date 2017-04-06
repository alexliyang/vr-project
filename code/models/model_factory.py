import os
from keras import backend as K
from keras.utils.visualize_util import plot
from metrics.metrics import cce_flatt, IoU, YOLOLoss, YOLOMetrics, MultiboxLoss, SSDMetrics
from models.densenetFCN import build_densenetFCN
from models.fcn8 import build_fcn8
from models.model import One_Net_Model
from models.resnet import build_resnet50
from models.ssd300 import build_ssd300
from models.vgg import build_vgg
from models.yolo import build_yolo
from models.deeplabV2 import build_deeplabv2
from models.segnet import build_segnet
from models.tiramisu import build_tiramisu_fc56, build_tiramisu_fc67, build_tiramisu_fc103

"""
from models.lenet import build_lenet
from models.alexNet import build_alexNet
from models.inceptionV3 import build_inceptionV3
from models.unet import build_unet
from models.resnetFCN import build_resnetFCN
from models.adversarial_semseg import Adversarial_Semseg
"""


# Build the model
class Model_Factory():
    def __init__(self):
        pass

    # Define the input size, loss and metrics
    def basic_model_properties(self, cf, variable_input_size):
        # Define the input size, loss and metrics
        if cf.dataset.class_mode == 'categorical':
            if K.image_dim_ordering() == 'th':
                in_shape = (cf.dataset.n_channels,
                            cf.target_size_train[0],
                            cf.target_size_train[1])
            else:
                in_shape = (cf.target_size_train[0],
                            cf.target_size_train[1],
                            cf.dataset.n_channels)
            loss = 'categorical_crossentropy'
            metrics = ['accuracy']
        elif cf.dataset.class_mode == 'detection':

            # Check model, different detection nets may have different losses and metrics
            if cf.model_name in ['yolo', 'tiny-yolo']:
                in_shape = (cf.dataset.n_channels,
                            cf.target_size_train[0],
                            cf.target_size_train[1])
                loss = YOLOLoss(in_shape, cf.dataset.n_classes, cf.dataset.priors)
                metrics = [YOLOMetrics(in_shape, cf.dataset.n_classes, cf.dataset.priors)]
            elif cf.model_name == 'ssd300':
                in_shape = (cf.target_size_train[0],
                            cf.target_size_train[1], cf.dataset.n_channels)

                loss = MultiboxLoss(cf.dataset.n_classes, neg_pos_ratio=2.0).compute_loss
                metrics = None
                # TODO: Add metrics for SSD
                # priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
                # metrics = [SSDMetrics(priors, cf.dataset.n_classes)]
            else:
                raise NotImplementedError

        elif cf.dataset.class_mode == 'segmentation':
            if K.image_dim_ordering() == 'th':
                in_shape = (cf.dataset.n_channels,
                            cf.target_size_train[0],
                            cf.target_size_train[1])
            else:
                in_shape = (cf.target_size_train[0],
                            cf.target_size_train[1],
                            cf.dataset.n_channels)
            loss = cce_flatt(cf.dataset.void_class, cf.dataset.cb_weights)
            metrics = [IoU(cf.dataset.n_classes, cf.dataset.void_class)]
        else:
            raise ValueError('Unknown problem type')
        return in_shape, loss, metrics

    # Creates a Model object (not a Keras model)
    def make(self, cf, optimizer=None):
        if cf.model_name in ['lenet', 'alexNet', 'vgg16', 'vgg19', 'resnet50',
                             'InceptionV3', 'resnetFCN', 'densenetFCN',
                             'yolo', 'tiny-yolo', 'ssd300',
                             'segnet_vgg', 'segnet_basic', 'fcn8', 'unet', 'deeplabV2',
                             'tiramisu_fc56', 'tiramisu_fc67', 'tiramisu_fc103']:
            if optimizer is None:
                raise ValueError('optimizer can not be None')

            # Fix input size at build time for Tiramisu network,
            # given the limitation of transposed convolutions when it comes to duplication of dimensions
            if 'tiramisu' in cf.model_name:
                variable_input_size = False
            else:
                variable_input_size = True
            in_shape, loss, metrics = self.basic_model_properties(cf, variable_input_size)
            model = self.make_one_net_model(cf, in_shape, loss, metrics,
                                            optimizer)

        elif cf.model_name == 'adversarial_semseg':
            if optimizer is None:
                raise ValueError('optimizer is not None')

            # loss, metrics and optimizer are made in class Adversarial_Semseg
            in_shape, _, _ = self.basic_model_properties(cf, False)
            model = Adversarial_Semseg(cf, in_shape)

        else:
            raise ValueError('Unknown model name')

        # Output the model
        print ('   Model: ' + cf.model_name)
        return model

    # Creates, compiles, plots and prints a Keras model. Optionally also loads its
    # weights.
    def make_one_net_model(self, cf, in_shape, loss, metrics, optimizer):
        # Create the *Keras* model
        if cf.model_name == 'fcn8':
            model = build_fcn8(in_shape, cf.dataset.n_classes, cf.weight_decay,
                               freeze_layers_from=cf.freeze_layers_from, path_weights=cf.load_imageNet)
        elif cf.model_name == 'segnet_basic':
            model = build_segnet(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                 freeze_layers_from=cf.freeze_layers_from,
                                 path_weights=None, basic=True)
        elif cf.model_name == 'segnet_vgg':
            model = build_segnet(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                 freeze_layers_from=cf.freeze_layers_from,
                                 path_weights=None, basic=False)
        elif cf.model_name == 'densenetFCN':
            model = build_densenetFCN(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                      freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'vgg16':
            model = build_vgg(in_shape, cf.dataset.n_classes, 16, cf.weight_decay,
                              load_pretrained=cf.load_imageNet,
                              freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'vgg19':
            model = build_vgg(in_shape, cf.dataset.n_classes, 19, cf.weight_decay,
                              load_pretrained=cf.load_imageNet,
                              freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'resnet50':
            model = build_resnet50(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                   load_pretrained=cf.load_imageNet,
                                   freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'yolo':
            model = build_yolo(in_shape, cf.dataset.n_classes,
                               cf.dataset.n_priors,
                               load_pretrained=cf.load_imageNet,
                               freeze_layers_from=cf.freeze_layers_from, tiny=False)
        elif cf.model_name == 'tiny-yolo':
            model = build_yolo(in_shape, cf.dataset.n_classes,
                               cf.dataset.n_priors,
                               load_pretrained=cf.load_imageNet,
                               freeze_layers_from=cf.freeze_layers_from, tiny=True)
        elif cf.model_name == 'ssd300':
            model = build_ssd300(in_shape, cf.dataset.n_classes + 1, cf.weight_decay,
                                 load_pretrained=cf.load_imageNet, freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'deeplabV2':
            model = build_deeplabv2(in_shape, classes=cf.dataset.n_classes, load_pretrained=cf.load_imageNet,
                                    freeze_layers_from=cf.freeze_layers_from, weight_decay=cf.weight_decay)
        elif cf.model_name == 'ssd300':
            model = build_ssd300(in_shape, cf.dataset.n_classes + 1, cf.weight_decay,
                                 load_pretrained=cf.load_imageNet,
                                 freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'tiramisu_fc56':
            model = build_tiramisu_fc56(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                        compression=0, dropout=0.2, nb_filter=48,
                                        freeze_layers_from=cf.freeze_layers_from)
        elif cf.model_name == 'tiramisu_fc67':
            model = build_tiramisu_fc67(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                        compression=0, dropout=0.2, nb_filter=48,
                                        freeze_layers_from=cf.freeze_layers_from)

        elif cf.model_name == 'tiramisu_fc103':
            model = build_tiramisu_fc103(in_shape, cf.dataset.n_classes, cf.weight_decay,
                                         compression=0, dropout=0.2, nb_filter=48,
                                         freeze_layers_from=cf.freeze_layers_from)

        else:
            raise ValueError('Unknown model')

        # Load pretrained weights
        if cf.load_pretrained:
            print('   loading model weights from: ' + cf.weights_file + '...')
            model.load_weights(cf.weights_file, by_name=True)

        # Compile model
        model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

        # Show model structure
        if cf.show_model:
            model.summary()
            plot(model, to_file=os.path.join(cf.savepath, 'model.png'))

        # Output the model
        print ('   Model: ' + cf.model_name)
        # model is a keras model, Model is a class wrapper so that we can have
        # other models (like GANs) made of a pair of keras models, with their
        # own ways to train, test and predict
        return One_Net_Model(model, cf, optimizer)
