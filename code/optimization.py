import subprocess
import os
import time


def create_config(newconfig_name, problem_type, dataset_name = None, model_name = None, batch_size_train = None,
                  optimizer = None, learning_rate = None, patience = None, n_epochs = None):
    if problem_type == 'classification':
        config_original = open('config/tt100k_classif.py', 'r')
    elif problem_type == 'segmentation':
        config_original = open('config/camvid_segmentation.py', 'r')

    config_new = open(newconfig_name, 'w')
    for line in config_original.readlines():

        if dataset_name is not None and 'dataset_name' in line:
            if problem_type == 'classification':
                line = line.replace('TT100K_trafficSigns', dataset_name, 1)
            elif problem_type == 'segmentation':
                line = line.replace('camvid', dataset_name, 1)
        elif model_name is not None and 'model_name' in line:
            if problem_type == 'classification':
                line = line.replace('vgg16', model_name, 1)
            elif problem_type == 'segmentation':
                line = line.replace('fcn8', model_name, 1)
        elif batch_size_train is not None and 'batch_size_train' in line:
            if problem_type == 'classification':
                line = line.replace('10', str(batch_size_train), 1)
            elif problem_type == 'segmentation':
                line = line.replace('5', str(batch_size_train), 1)
        elif optimizer is not None and 'optimizer' in line:
            line = line.replace('rmsprop', optimizer)
        elif learning_rate is not None and 'learning_rate' in line:
            line = line.replace('0.0001' , str(learning_rate), 1)
        elif patience is not None and 'earlyStopping_patience' in line:
            line = line.replace('100', str(patience), 1)
        elif n_epochs is not None and 'n_epochs' in line:
            if problem_type == 'classification':
                line = line.replace('30', str(n_epochs), 1)
            elif problem_type == 'segmentation':
                line = line.replace('30', str(n_epochs), 1)

        config_new.write(line+'\n')
    config_new.close()
    config_original.close()




learning_rate = [0.00001, 0.0001]
optimizer = ['adam']
batch_size_train = [20]
problem_type = 'classification'
#dataset = 'TT100K_trafficSigns'
i = 0
for lr in learning_rate:
    for opt in optimizer:
        for bsz in batch_size_train:
            config_name = 'config/config'+str(i)+'.py'
            create_config(config_name, problem_type, batch_size_train=bsz, optimizer=opt, learning_rate=lr)
            while not os.path.isfile(config_name):
                time.sleep(1)
            subprocess.call(['python', 'train.py', '-c', config_name, '-e', 'optimization_lr_{}_batchsizetrain_{}_'
                                                                            'opt_{}'.format(lr, bsz, opt)])

