import os
import subprocess
import time


def create_config(newconfig_name, problem_type, dataset_name=None, model_name=None, batch_size_train=None,
                  batch_size_test=None, optimizer=None, learning_rate=None, patience=None, n_epochs=None,
                  es_monitor=None, es_mode=None, debug=None, weight_decay = None, lr_decay_factor = None,
                  lr_decay_epochs = None, lr_decay_enable = None):
    # Config file for classification
    if problem_type == 'classification':
        config_original_filename = 'config/tt100k_classif.py'
    # Config file for segmentation
    elif problem_type == 'segmentation':
        config_original_filename = 'config/camvid_segmentation.py'
    else:
        raise Exception('{} is not supported. Only classification and segmentation problems are supported so far.')

    with open(config_original_filename, 'rb') as config_original, open(newconfig_name, 'wb') as config_new:
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
            elif batch_size_test is not None and 'batch_size_test' in line:
                if problem_type == 'classification':
                    line = line.replace('30', str(batch_size_test), 1)
                elif problem_type == 'segmentation':
                    line = line.replace('5', str(batch_size_test), 1)
            elif optimizer is not None and 'optimizer' in line:
                line = line.replace('rmsprop', optimizer)
            elif learning_rate is not None and 'learning_rate' in line:
                line = line.replace('0.0001', str(learning_rate), 1)
            elif patience is not None and 'earlyStopping_patience' in line:
                line = line.replace('100', str(patience), 1)
            elif n_epochs is not None and 'n_epochs' in line:
                if problem_type == 'classification':
                    line = line.replace('30', str(n_epochs), 1)
                elif problem_type == 'segmentation':
                    line = line.replace('30', str(n_epochs), 1)
            elif 'show_model =' in line:
                line = line.replace('False', 'True', 1)
            elif 'earlyStopping_monitor' in line and es_monitor is not None:
                line = line.replace('acc', es_monitor, 1)
            elif 'earlyStopping_mode' in line and es_mode is not None:
                line = line.replace('max', es_mode, 1)
            elif debug is not None and 'debug' in line:
                line = line.replace('False', str(debug), 1)
            elif weight_decay is not None and 'weight_decay' in line:
                line = line.replace('0.', str(weight_decay), 1)
            elif lr_decay_epochs is not None and 'lrDecayScheduler_epochs' in line:
                line = line.replace('[5, 10, 20]', str(lr_decay_epochs), 1)
            elif lr_decay_factor is not None and 'lrDecayScheduler_rate' in line:
                line = line.replace('2', str(lr_decay_factor), 1)
            elif lr_decay_enable is not None and 'lrDecayScheduler_enabled' in line:
                line = line.replace('False', str(lr_decay_enable), 1)
            # Write new line
            config_new.write(line)


if __name__ == '__main__':

    learning_rates = [0.001, 0.0001]
    optimizers = ['adam', 'rmsprop']
    weight_decay = [0.001, 0.0001]

    batch_sizes_train = 30
    batch_sizes_test = 30
    prob_type = 'classification'
    debug = False
    model = 'resnet50'
    lr_decay_factor = 10
    lr_decay_epochs = [15, 25]
    lr_decay_enable = True
    i = 0
    for lr in learning_rates:
        for opt in optimizers:
            for wd in weight_decay:
                config_name = '{}_optimization_lr_{}_wd_{}_opt_{}.py'.format(model, lr, wd, opt)
                create_config(config_name, prob_type, model_name=model, weight_decay = wd,
                              optimizer=opt, learning_rate=lr, lr_decay_factor = lr_decay_factor,
                              lr_decay_epochs= lr_decay_epochs, batch_size_train=batch_sizes_train, batch_size_test=batch_sizes_test,
                              lr_decay_enable = lr_decay_enable, debug=debug)
                while not os.path.isfile(config_name):
                    time.sleep(1)
                subprocess.call(
                    ['python', 'train.py', '-c', config_name, '-e', '{}_optimization_lr_{}_wd_{}_opt_{}.py'.format(model, lr, wd, opt)])
