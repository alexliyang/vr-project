from __future__ import print_function
import pandas as pd
import argparse
from glob import glob
import os

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

        # Iterate over all sets (train, val and test)
        for set_type_path in glob(os.path.join(dataset, '*')):
            if os.path.isdir(set_type_path):
                set_type = os.path.basename(set_type_path)
                analysis_res[set_type] = {}
                # Iterate over all available classes
                for category in glob(os.path.join(set_type_path, '*')):
                    if os.path.isdir(category):
                        category_name = os.path.basename(category)
                        # Get the number of images in this directory (dataset/set_type/category)
                        num_elems = len(os.listdir(category))
                        # Put it in the results placeholder
                        analysis_res[set_type].update({category_name: num_elems})

        analysis_file_path = os.path.join(output_path, '{}_analysis.csv'.format(dataset_name))
        dataframe = pd.DataFrame(analysis_res)
        dataframe.to_csv(analysis_file_path)
        print('Analysis finished\n')
