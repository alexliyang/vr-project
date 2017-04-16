from __future__ import print_function, division

import os
import time
import pickle
import argparse

from keras.preprocessing import image

from models.yolo import build_yolo
from models.ssd300 import build_ssd300
from tools.yolo_utils import *
from tools.ssd_utils import BBoxUtility
import matplotlib.pyplot as plt

plt.switch_backend('Agg')
plt.rcParams['image.interpolation'] = 'nearest'

""" MAIN SCRIPT """

if __name__ == '__main__':

    """ CONSTANTS """
    available_models = ['yolo', 'tiny-yolo', 'ssd']
    available_datasets = {
        'TT100K_detection': [
            'i2', 'i4', 'i5', 'il100', 'il60', 'il80', 'io', 'ip', 'p10', 'p11', 'p12', 'p19', 'p23', 'p26', 'p27',
            'p3', 'p5', 'p6', 'pg', 'ph4', 'ph4.5', 'ph5', 'pl100', 'pl120', 'pl20', 'pl30', 'pl40', 'pl5', 'pl50',
            'pl60', 'pl70', 'pl80', 'pm20', 'pm30', 'pm55', 'pn', 'pne', 'po', 'pr40', 'w13', 'w32', 'w55', 'w57',
            'w59', 'wo'
        ],
        'Udacity': ['Car', 'Pedestrian', 'Truck']
    }
    input_shape = (3, 320, 320)
    image_width = input_shape[2]
    image_height = input_shape[1]
    priors = [[0.9, 1.2], [1.05, 1.35], [2.15, 2.55], [3.25, 3.75], [5.35, 5.1]]
    prediction_images_dir = 'prediction_images'

    """ PARSE ARGS """
    arguments_parser = argparse.ArgumentParser()

    arguments_parser.add_argument('model', help='Model name', choices=available_models)
    arguments_parser.add_argument('dataset', help='Name of the dataset', choices=available_datasets.keys())
    arguments_parser.add_argument('weights', help='Path to the weights file')
    arguments_parser.add_argument('test_folder', help='Path to the folder with the images to be tested')
    # IMPORTANT: the values of these two params will affect the final performance of the network
    arguments_parser.add_argument('--detection-threshold', help='Detection threshold (between 0 and 1)',
                                  default=0.5, type=float)
    arguments_parser.add_argument('--nms-threshold', help='Non-maxima supression threshold (between 0 and 1)',
                                  default=0.2, type=float)
    arguments_parser.add_argument('--ignore-class', help='List of classes to be ignore from predictions',
                                  type=int,
                                  nargs='+')

    arguments = arguments_parser.parse_args()

    model_name = arguments.model
    dataset_name = arguments.dataset
    weights_path = arguments.weights
    test_dir = arguments.test_folder
    dataset_split_name = test_dir.split('/')[-2]
    detection_threshold = arguments.detection_threshold
    nms_threshold = arguments.nms_threshold
    ignore_class = arguments.ignore_class or []

    # Create directory to store predictions
    try:
        os.makedirs(prediction_images_dir)
    except Exception:
        pass

    # Classes for this dataset
    classes = available_datasets[dataset_name]
    num_classes = len(classes)
    if 'ssd' in model_name:
        num_classes += 1  # Background class included

    # Ignored classes
    if ignore_class:
        list_ignored_classes = ', '.join([classes[i] for i in ignore_class])
        print()
        print('IGNORED CLASSES: {}'.format(list_ignored_classes))
        print()

    # Plotting options
    colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()
    plt.ioff()

    # Create the model
    if model_name == 'tiny-yolo':
        model = build_yolo(img_shape=input_shape, n_classes=num_classes, n_priors=5,
                           load_pretrained=False, freeze_layers_from='base_model',
                           tiny=True)

    elif model_name == 'ssd':
        input_shape_ssd = np.roll(input_shape, -1)
        model = build_ssd300(input_shape_ssd.tolist(), num_classes, 0,
                             load_pretrained=False,
                             freeze_layers_from='base_model')
    elif model_name == 'yolo':
        model = build_yolo(img_shape=input_shape, n_classes=num_classes, n_priors=5,
                           load_pretrained=False, freeze_layers_from='base_model',
                           tiny=False)
    else:
        print("Error: Model not supported!")
        exit(1)

    # Load weights
    model.load_weights(weights_path)

    # Get images from test directory
    imfiles = [os.path.join(test_dir, f) for f in os.listdir(test_dir)
               if os.path.isfile(os.path.join(test_dir, f))
               and f.endswith('jpg')]

    if len(imfiles) == 0:
        print("ERR: path_to_images does not contain any jpg file")
        exit(1)

    inputs = []
    img_paths = []
    chunk_size = 128  # we are going to process all image files in chunks

    ok = 0.
    total_true = 0.
    total_pred = 0.

    mean_fps = 0.
    iterations = 0.

    for i, img_path in enumerate(imfiles):
        img = image.load_img(img_path, target_size=(input_shape[1], input_shape[2]))
        img = image.img_to_array(img)
        img /= 255.
        inputs.append(img.copy())
        img_paths.append(img_path)

        if len(img_paths) % chunk_size == 0 or i + 1 == len(imfiles):
            iterations += 1
            inputs = np.array(inputs)
            start_time = time.time()
            net_out = model.predict(inputs, batch_size=16, verbose=1)
            sec = time.time() - start_time
            fps = len(inputs) / sec
            print('{} images predicted in {:.5f} seconds. {:.5f} fps'.format(len(inputs), sec, fps))

            # find correct detections (per image)
            for i, img_path in enumerate(img_paths):
                if model_name == 'yolo' or model_name == 'tiny-yolo':
                    boxes_pred = yolo_postprocess_net_out(net_out[i], priors, classes, detection_threshold,
                                                          nms_threshold)
                elif model_name == 'ssd':
                    priors = pickle.load(open('prior_boxes_ssd300.pkl', 'rb'))
                    real_num_classes = num_classes - 1  # Background is not included
                    bbox_util = BBoxUtility(real_num_classes, priors=priors, nms_thresh=nms_threshold)
                    boxes_pred = bbox_util.detection_out(net_out[i],
                                                         background_label_id=0,
                                                         confidence_threshold=detection_threshold)
                else:
                    print("Error: Model not supported!")
                    exit(1)

                # Plot first image
                if i == 0:
                    current_img = inputs[0]
                    if 'yolo' in model_name:
                        current_img = np.transpose(current_img, (1, 2, 0))
                    plt.imshow(current_img)
                    currentAxis = plt.gca()

                # Compute number of predictions that match with GT with a minimum of 50% IoU
                for b in boxes_pred:
                    pred_idx = np.argmax(b.probs)

                    # Do not count as prediction if it is below the detection threshold or in the ignore list
                    if (b.probs[pred_idx] < detection_threshold) or (pred_idx in ignore_class):
                        continue

                    total_pred += 1.

                    if i == 0:
                        # Plot current prediction
                        xmin = int(round((b.x - b.w / 2) * image_width))
                        ymin = int(round((b.y - b.h / 2) * image_height))
                        xmax = int(round((b.x + b.w / 2) * image_width))
                        ymax = int(round((b.y + b.h / 2) * image_height))
                        score = b.c
                        label_idx = int(np.argmax(b.probs))
                        label = classes[label_idx]
                        display_txt = '{:0.2f}, label={}'.format(score, label)
                        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
                        color = colors[label_idx]
                        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=1))
                        currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.6})

                if i == 0:
                    plt.savefig('{}/{}_{}_{}_{}.png'.format(
                        prediction_images_dir,
                        model_name,
                        dataset_name,
                        dataset_split_name,
                        int(iterations)
                    ))
                    plt.close()

            inputs = []
            img_paths = []
