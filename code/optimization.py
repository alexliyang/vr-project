import subprocess
import os
import time


def create_config(newconfig_name, problem_type, dataset_name = None, model_name = None, batch_size_train = None,
                  optimizer = None, learning_rate = None):
    if problem_type == 'classification':
        config_original = open('config/tt100k_classif.py', 'r')
    elif problem_type == 'segmentation':
        config_original = open('config/camvid_segmentation.py', 'r')

    config_new = open('config/'+newconfig_name, 'w')
    for line in config_original.readlines():

        if dataset_name is not None and 'dataset_name' in line:
            if problem_type == 'classification':
                line.replace('TT100K_trafficSigns', dataset_name)
            elif problem_type == 'segmentation':
                line.replace('camvid', dataset_name)
        elif model_name is not None and 'model_name' in line:
            if problem_type == 'classification':
                line.replace('vgg16', model_name)
            elif problem_type == 'segmentation':
                line.replace('fcn8', model_name)
        elif batch_size_train is not None and 'batch_size_train' in line:
            if problem_type == 'classification':
                line.replace('10', batch_size_train)
            elif problem_type == 'segmentation':
                line.replace('5', batch_size_train)
        elif optimizer is not None and 'optimizer' in line:
            line.replace('rmsprop', optimizer)
        elif learning_rate is not None and 'learning_rate' in line:
            line.replace('0.0001' , learning_rate)

        config_new.writelines(line)




learning_rate = [0.00001, 0.0001]
optimizer = ['adam']
batch_size_train = [10, 15, 20]
problem_type = 'classification'
#dataset = 'TT100K_trafficSigns'
i = 0
for lr in learning_rate:
    for opt in optimizer:
        for bsz in batch_size_train:
            config_name = 'config'+str(i)+'.py'
            create_config(config_name, problem_type, batch_size_train=bsz, optimizer=opt, learning_rate=lr)
            while not os.path.isfile('config/'+config_name):
                time.sleep(1)
            subprocess.call(['python', 'train.py', '-c', config_name, '-e', 'optimization_lr_{}_batchsizetrain_{}_'
                                                                            'opt_{}'.format(lr, bsz, opt)])

