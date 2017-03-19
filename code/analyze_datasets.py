from __future__ import print_function, division

import argparse
from glob import glob

import pandas as pd
import seaborn
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.ioff()

# Maximum width for the bar plot
MAX_WIDTH = 0.8
COLORS = ('red', 'blue', 'yellow', 'green', 'purple')


def plot_bar_chart(ds_name, histogram_dict, save_path, normalized=True):
    """
    Plot the distribution of a dataset given a dictionary with the distribution of classes per
    data split (train, validation and test)

    :param ds_name: Name of the dataset being analyzed
    :type ds_name: str
    :param histogram_dict: Dictionary with keys being the dataset split and values the distributions of samples
    per class
    :type histogram_dict: dict
    :param save_path: Path where the plot will be saved
    :type save_path: str
    :param normalized: Whether the distribution is normalized (sums 1) or not
    :type normalized: bool
    """
    # Compute the width of each bar
    width = MAX_WIDTH / len(histogram_dict)

    # Iterate over all sets
    plt.figure(figsize=(20, 10), dpi=150)
    legend_axes = []
    legend_names = []
    for ind, (set_type, distribution) in enumerate(histogram_dict.iteritems()):
        num_classes = len(distribution)
        sum_distribution = np.sum(distribution)
        if num_classes > 0 and sum_distribution > 0:
            indexes = np.arange(num_classes)
            if normalized:
                distribution = np.array(distribution) / sum_distribution
            else:
                distribution = np.array(distribution)
            ax_res = plt.bar(indexes + ind * width, distribution, width, color=COLORS[ind])
            legend_axes.append(ax_res[0])
            legend_names.append(set_type)

    if normalized:
        plt.ylabel('Normalized count')
    else:
        plt.ylabel('Count')
    plt.title(ds_name.upper())

    plt.legend(tuple(legend_axes), tuple(legend_names))
    plt.savefig(save_path)


if __name__ == '__main__':
    program_description = '''Analyzes the number of images per class in all the datasets of a specific problem
    type (classification, detection, segmentation) in the
    specified path and stores the analysis as CSV files.
    The following are valid structures for the different problem types:

    CLASSIFICATION
        - dataset
            - train
                - class1
                    - image1.png
                    - image2.png
                    - (...)
                - class2
                    - image45.png
                    - (...)
                - (...)
            - val
                - class 1
                    - (...)
                - (...)
            - test
                - class 1
                        - (...)
                    - (...)

    DETECTION
        - dataset
            - train
                - image1.png
                - image1.txt
                - image2.png
                - image2.txt
                - (...)
                - image45.png
                - image45.txt
                - (...)
            - val
                - (...)
            - test
                - (...)

    SEGMENTATION
        - dataset
            - train
                - images
                    - image1.png
                    - image2.png
                - masks
                    - image1.png
                    - image2.png
            - val
                - images
                    - (...)
                - masks
                    - (...)
            - test
                - images
                    - (...)
                - masks
                    - (...)
    '''
    argparser = argparse.ArgumentParser(description=program_description)
    argparser.add_argument('type', choices=['classification', 'detection', 'segmentation'], help='Dataset type')
    argparser.add_argument('data', help='Datasets path')
    argparser.add_argument('--output', help='Path where the csv files will be stored')

    arguments = argparser.parse_args()
    dataset_type = arguments.type
    data_path = arguments.data
    data_path = os.path.join(data_path, dataset_type)
    output_path = arguments.output

    # If output path is not defined, use data path
    if output_path is None:
        output_path = data_path

    # Iterate over all datasets
    for dataset in glob(os.path.join(data_path, '*')):
        dataset_name = os.path.basename(dataset)
        print('{:-^40}'.format(''))
        print('{:-^40}'.format('Analyzing {}'.format(dataset_name.upper())))
        print('{:-^40}'.format(''))
        print('Path: {}'.format(dataset))

        # Placeholder to store the results of the analysis for this dataset
        # Not all of them are used, as some are specific to problem types (e.g. aspect ratio to detection)
        analysis_res = {}
        analysis_hist = {}
        aspect_ratios = {}
        bb_areas = {}

        if os.path.isdir(dataset):

            # Iterate over all sets (train, val and test)
            for set_type_path in glob(os.path.join(dataset, '*')):

                if os.path.isdir(set_type_path):
                    # Data split name (train, validation, test)
                    set_type = os.path.basename(set_type_path)

                    # Variable to store the distribution of classes
                    analysis_res[set_type] = {}

                    # Classification
                    if dataset_type == 'classification':

                        # Variable to store the distribution of classes (as an array)
                        analysis_hist[set_type] = []

                        # Iterate over all available classes
                        for category in glob(os.path.join(set_type_path, '*')):
                            if os.path.isdir(category):
                                category_name = os.path.basename(category)
                                # Get the number of images in this directory (dataset/set_type/category)
                                num_elems = len(os.listdir(category))
                                # Put it in the results placeholder
                                analysis_res[set_type].update({category_name: num_elems})
                                analysis_hist[set_type].append(num_elems)

                    # Detection
                    elif dataset_type == 'detection':

                        # Variable to store the distribution of aspect ratios
                        aspect_ratios[set_type] = []
                        bb_areas[set_type] = []

                        # Iterate over all annotation files
                        for annotation in glob(os.path.join(set_type_path, '*.txt')):
                            # Read the file, and for each instance of a class, store it in the results dictionaries
                            f = open(annotation, mode='rb')
                            for line in f.readlines():
                                # Instance class
                                class_id = int(line.split(' ')[0])

                                # Bounding box width and height
                                width, height = map(lambda x: float(x.replace('\n', '')), line.split(' ')[3:])

                                # Bounding box aspect ratio
                                aspect_r = width / height
                                aspect_ratios[set_type].append(aspect_r)

                                # Bounding box area
                                bb_area = width * height
                                bb_areas[set_type].append(bb_area)

                                # Update the distribution over classes
                                try:
                                    analysis_res[set_type][class_id] += 1
                                except KeyError:
                                    analysis_res[set_type][class_id] = 1
                            f.close()

                    # Segmentation
                    elif dataset_type == 'segmentation':
                        raise NotImplementedError

                    # Another
                    else:
                        raise ValueError('Dataset type must be one of the following: "classification", "detection", '
                                         '"segmentation".')

            # Create CSV file with results
            analysis_file_path = os.path.join(output_path, '{}_class_analysis.csv'.format(dataset_name))
            dataframe = pd.DataFrame(analysis_res)
            dataframe.to_csv(analysis_file_path)

            # Non normalized distribution
            plt.figure(figsize=(20, 10))
            dataframe.plot.bar()
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, '{}_class_distribution.png'.format(dataset_name)))

            # Normalized distribution
            plt.figure(figsize=(20, 10))
            norm_dataframe = dataframe / dataframe.sum(axis=0)
            norm_dataframe.plot.bar()
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, '{}_class_distribution_norm.png'.format(dataset_name)))

            # Detection specific plots
            if dataset_type == 'classification':
                # Plot bar chart with distributions for all sets
                plot_bar_chart(
                    dataset_name,
                    analysis_hist,
                    os.path.join(output_path, '{}_distribution_norm.png'.format(dataset_name)),
                    normalized=True
                )
                plot_bar_chart(
                    dataset_name,
                    analysis_hist,
                    os.path.join(output_path, '{}_distribution.png'.format(dataset_name)),
                    normalized=False
                )

            if dataset_type == 'detection':
                # Plot histogram of aspect ratios per data split
                plt.figure(figsize=(10, 10))
                for set_name, ar_values in aspect_ratios.iteritems():
                    plt.hist(ar_values, bins=100, normed=True, label=set_name, range=(0, 4))
                plt.tight_layout()
                plt.legend(loc='best')
                plt.savefig(os.path.join(output_path, '{}_aspect_ratios_hist.png'.format(dataset_name)))

                # Plot histogram of areas per data split
                plt.figure(figsize=(10, 10))
                for set_name, bb_ar_values in bb_areas.iteritems():
                    plt.hist(bb_ar_values, bins=100, normed=True, label=set_name)
                plt.tight_layout()
                plt.legend(loc='best')
                plt.savefig(os.path.join(output_path, '{}_bb_areas_hist.png'.format(dataset_name)))

        print('Analysis finished\n')
