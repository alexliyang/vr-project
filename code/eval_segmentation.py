from __future__ import print_function, division

import os
import numpy as np
import argparse
import glob

import keras.backend as K
from keras.preprocessing import image

from skimage import data

from models.fcn8 import build_fcn8
from models.segnet import build_segnet
from models.deeplabV2 import build_deeplabv2
from models.dilation import build_dilation
from models.tiramisu import build_tiramisu_fc56, build_tiramisu_fc67, build_tiramisu_fc103

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.ioff()

""" MAIN SCRIPT """

if __name__ == '__main__':

    """ CONSTANTS """
    available_models = {
        'fcn8': build_fcn8,
        'segnet': build_segnet,
        'deeplabv2': build_deeplabv2,
        'dilated': build_dilation,
        'tiramisu_fc56': build_tiramisu_fc56,
        'tiramisu_fc67': build_tiramisu_fc67,
        'tiramisu_fc103': build_tiramisu_fc103
    }
    available_datasets = {
        'camvid': {
            # TODO: Camvid classes dict
        },
        'cityscapes': {
            # TODO: Cityscapes classes dict
        },
    }
    home_dir = os.path.expanduser('~')
    dim_ordering = K.image_dim_ordering()
    channel_axis = 3 if dim_ordering == 'tf' else 1
    chunk_size = 128

    """ PARSE ARGS """
    arguments_parser = argparse.ArgumentParser()

    arguments_parser.add_argument('model', help='Model name', choices=available_models)
    arguments_parser.add_argument('dataset', help='Name of the dataset', choices=available_datasets.keys())
    arguments_parser.add_argument('weights', help='Path to the weights file')
    arguments_parser.add_argument('test', help='Path to the folder with the images to be tested')
    arguments = arguments_parser.parse_args()

    model_name = arguments.model
    dataset = arguments.dataset
    weights_path = arguments.weights
    test_dir = arguments.test

    # Create directory to store predictions
    try:
        predictions_folder = os.path.join(home_dir, 'prediction-{}-{}-{}'.format(
            model_name,
            dataset,
            os.path.basename(test_dir)
        ))
        os.makedirs(predictions_folder)
    except OSError:
        # Ignore
        pass

    # Classes and number of classes
    seg_classes = available_datasets[dataset]
    num_classes = len(seg_classes)

    # Images to be predicted
    test_images = glob.glob(os.path.join(test_dir, '*.png'))
    test_images += glob.glob(os.path.join(test_dir, '*.jpg'))
    total_images = len(test_images)

    # Input shape (get it from first image)
    sample_image_path = test_images[0]
    sample_image = data.load(sample_image_path)
    input_shape = sample_image.shape

    if dim_ordering == 'th':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    # Create the model
    model = available_models[model_name](img_shape=input_shape, nclasses=num_classes)

    # Load weights
    model.load_weights(weights_path)

    iteration = 0

    for i in range(0, test_images, chunk_size):
        chunked_img_list = test_images[i:i + chunk_size]
        images = np.array([image.img_to_array(image.load_img(f)) / 255. for f in chunked_img_list])
        # FIXME: Remove this print used for debugging
        print(images.shape)

        pred = model.predict(images, batch_size=2)
        y_pred = np.argmax(pred, axis=channel_axis)

        # TODO: Store the predictions

