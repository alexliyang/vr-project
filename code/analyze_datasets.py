from __future__ import print_function, division

import argparse
from glob import glob

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
plt.ioff()

# Maximum width for the bar plot
MAX_WIDTH = 0.8
COLORS = ('red', 'blue', 'yellow', 'green', 'purple')


def plot_bar_chart(dataset_name, histogram_dict, save_path, normalized=True):
    # Compute the width of each bar
    width = MAX_WIDTH / len(histogram_dict)

    # Iterate over all sets
    plt.figure(figsize=(20, 10), dpi=150)
    legend_axes = []
    legend_names = []
    for ind, (set_type, distribution) in enumerate(analysis_hist.iteritems()):
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
    plt.title(dataset_name.upper())

    plt.legend(tuple(legend_axes), tuple(legend_names))
    plt.savefig(save_path)


if __name__ == '__main__':
    program_description = '''Analyzes the number of images per class in all the classification datasets in the
    specified path and stores the analysis as CSV files.
    It is assumed that a dataset has the following folder structure:
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
    '''
    argparser = argparse.ArgumentParser(description=program_description)
    argparser.add_argument('data', help='Classification datasets path')
    argparser.add_argument('--output', help='Path where the csv files will be stored')

    arguments = argparser.parse_args()
    data_path = arguments.data
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
        analysis_res = {}
        analysis_hist = {}

        if os.path.isdir(dataset):
            # Iterate over all sets (train, val and test)
            for set_type_path in glob(os.path.join(dataset, '*')):
                if os.path.isdir(set_type_path):
                    set_type = os.path.basename(set_type_path)
                    analysis_res[set_type] = {}
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

            # Create CSV file with results
            analysis_file_path = os.path.join(output_path, '{}_analysis.csv'.format(dataset_name))
            dataframe = pd.DataFrame(analysis_res)
            dataframe.to_csv(analysis_file_path)

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

        print('Analysis finished\n')
