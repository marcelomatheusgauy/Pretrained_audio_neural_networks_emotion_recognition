import os
import sys
import numpy as np
import argparse
import h5py
import math
import random
import time
import logging
import matplotlib.pyplot as plt
import sklearn

import torch
import torchaudio
#torch.backends.cudnn.benchmark=True
#torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
sys.path.append("utils")
from utilities import get_filename
sys.path.append("pytorch/")
from models import *
import config



#load transformers and training functions
from train_utils import run_epoch, LossCompute, NoamOpt, Transfer_Cnn6, Transfer_Cnn10, Transfer_Cnn14

def run_training_function(output_filename, model_type, model_save_path, pretrain):

    output_file = open(output_filename, 'w')

    data_emotion_target_dictionary = {}

    #emotion recognition
    folder = '../../data_train/train/'

    data_paths_train = []

    for file in os.listdir(folder):
        file_path = folder + file
        #to be consistent with rest of the code - we do not want to split these into 4 seconds audios
        data_path = file_path
        data_paths_train.append(data_path)
        v = file.replace('.wav','').split('_')
        emotion_label = v[len(v)-1]
        data_emotion_target_dictionary[file_path] = emotion_label
        #print(emotion_label)

        random.shuffle(data_paths_train)
    
    number_files = len(data_paths_train)
    #using part of training set as test set
    #validation_cut = int(math.floor(0.8*number_files))
    #test_cut = int(math.floor(0.9*number_files))
    #data_paths_valid = data_paths_train[validation_cut:test_cut]
    #data_paths_test = data_paths_train[test_cut:]
    #data_paths_train = data_paths_train[:validation_cut]   

    #using official test set
    #validation_cut = int(math.floor(0.9*number_files))
    #data_paths_valid = data_paths_train[validation_cut:]
    #data_paths_train = data_paths_train[:validation_cut]


    folder = '../../data_train/validation/'

    data_paths_valid = []

    for file in os.listdir(folder):
        file_path = folder+file
        data_path = file_path
        data_paths_valid.append(data_path)
        v = file.replace('.wav','').split('_')
        emotion_label = v[len(v)-1]
        data_emotion_target_dictionary[file_path] = emotion_label

    random.shuffle(data_paths_valid)

    folder = '../../data_train/test/'

    data_paths_test = []

    for file in os.listdir(folder):
        file_path = folder+file
        data_path = file_path
        data_paths_test.append(data_path)
        v = file.replace('.wav','').split('_')
        emotion_label = v[len(v)-1]
        data_emotion_target_dictionary[file_path] = emotion_label

    random.shuffle(data_paths_test)
    
    
    #using official test set

    folder = '../../data_train/test_ser_labeled/'

    data_paths_test_official = []

    for file in os.listdir(folder):
        file_path = folder + file
        #to be consistent with rest of the code - we do not want to split these into 4 seconds audios
        data_path = file_path
        data_paths_test_official.append(data_path)
        v = file.replace('.wav','').split('_')
        emotion_label = v[len(v)-1]
        data_emotion_target_dictionary[file_path] = emotion_label
    

    ##############################################################################################

    ##############################################################################################

    #run training
    args = {}
    args['sample_rate']= 32000
    args['window_size']= 1024
    args['hop_size']=320
    args['mel_bins']=64
    args['fmin']=0
    args['fmax']=32000
    args['model_type']=model_type#"Transfer_Cnn10"
    #args['pretrained_checkpoint_path']=model_load_path#"Cnn10_mAP=0.380.pth"
    args['freeze_base']=False
    args['cuda']=True



    # Arguments & parameters
    sample_rate = args['sample_rate']
    window_size = args['window_size']
    hop_size = args['hop_size']
    mel_bins = args['mel_bins']
    fmin = args['fmin']
    fmax = args['fmax']
    model_type = args['model_type']
    #pretrained_checkpoint_path = args['pretrained_checkpoint_path']
    freeze_base = args['freeze_base']
    device = 'cuda' if (args['cuda'] and torch.cuda.is_available()) else 'cpu'
    classes_num = 3#config.classes_num
    #pretrain = True if pretrained_checkpoint_path else False

    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)


    # Load pretrained model
    #if pretrain:
    #    logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
    #    model.load_from_pretrain(pretrained_checkpoint_path)


    if 'cuda' in device:
        model.to(device)

    print('Load pretrained model successfully!', file=output_file)



    pretrain = 'emotion'
    d_model = 512
    model_opt = NoamOpt(d_model, 1, 400,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    avg_loss=0
    best_val_acc = 0
    best_val_f1_score = 0
    model_path = model_save_path
    min_frequency = 0.0
    max_frequency = None

    for epoch in range(100):
        model.train()
        loss, avg_loss, _, _, _ = run_epoch(model, 
                  LossCompute(model, model_opt, pretrain),
                  data_paths_train, data_emotion_target_dictionary, output_file, pretrain=pretrain, training=True, avg_loss=avg_loss,
                  min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
        model.eval()
        with torch.no_grad():
            loss, _, val_acc, val_f1_score, true_val_f1_score = run_epoch(model, 
                        LossCompute(model, None, pretrain=pretrain),
                        data_paths_valid, data_emotion_target_dictionary, output_file, pretrain=pretrain, training=False,
                        min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=64)
            print(val_acc, file=output_file)
            print(val_f1_score, file=output_file)
        #if best_val_acc < val_acc:
            #best_val_acc = val_acc
        if best_val_f1_score < true_val_f1_score:
            best_val_f1_score = true_val_f1_score
            print('Saving model', file=output_file)
            torch.save({
                'model_state_dict': model.state_dict()
                }, model_path)
            #torch.save(model, model_path)


    #model_path = 'model_test_mel.ckpt'
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, freeze_base)

    # Load trained model
    logging.info('Load pretrained model from {}'.format(model_path))
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'cuda' in device:
        model.to(device)
    V=64
    pretrain = 'emotion'
    model.eval()
    with torch.no_grad():
        print('End training. Check validation', file=output_file)
        print(run_epoch(model, 
                            LossCompute(model, None, pretrain=pretrain),
                            data_paths_valid, data_emotion_target_dictionary, output_file, pretrain=pretrain, training=False, number_coeffs=64), file=output_file)
        print('End validation. Check test', file=output_file)
        print(run_epoch(model, 
                            LossCompute(model, None, pretrain=pretrain),
                            data_paths_test, data_emotion_target_dictionary, output_file, pretrain=pretrain, training=False, number_coeffs=64), file=output_file)
        print('End test. Check official test', file=output_file)
        print(run_epoch(model, 
                            LossCompute(model, None, pretrain=pretrain),
                            data_paths_test_official, data_emotion_target_dictionary, output_file, pretrain=pretrain, training=False, number_coeffs=64), file=output_file)
