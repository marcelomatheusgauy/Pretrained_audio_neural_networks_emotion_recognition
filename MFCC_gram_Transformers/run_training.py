import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import sklearn.metrics
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import random
import os
import pandas


#load transformers and training functions
from transformer_encoder import make_transformer_encoder_model
from train_utils import run_epoch, LossCompute, NoamOpt, build_data_paths_with_index

def run_training_function(output_filename, extract_coeffs, pretrain, d_model, d_ff, dropout):

    output_file = open(output_filename, 'w')

    #
    data_emotion_target_dictionary = {}

    
    #emotion recognition
    folder = 'data_train/train/'

    data_paths_train = []

    for file in os.listdir(folder):
        file_path = folder + file
        #to be consistent with rest of the code - we do not want to split these into 4 seconds audios
        data_path = file_path + '_0'
        data_paths_train.append(data_path)
        v = file.replace('.wav','').split('_')
        emotion_label = v[len(v)-1]
        data_emotion_target_dictionary[file_path] = emotion_label
        #print(emotion_label)

    random.shuffle(data_paths_train)
    
    number_files = len(data_paths_train)

    folder = 'data_train/validation/'

    data_paths_valid = []

    for file in os.listdir(folder):
        file_path = folder+file
        data_path = file_path+'_0'
        data_paths_valid.append(data_path)
        v = file.replace('.wav','').split('_')
        emotion_label = v[len(v)-1]
        data_emotion_target_dictionary[file_path] = emotion_label

    random.shuffle(data_paths_valid)
    
    folder = 'data_train/test/'

    data_paths_test = []

    for file in os.listdir(folder):
        file_path = folder+file
        data_path = file_path + '_0'
        data_paths_test.append(data_path)
        v = file.replace('.wav','').split('_')
        emotion_label = v[len(v)-1]
        data_emotion_target_dictionary[file_path] = emotion_label

    random.shuffle(data_paths_test)
    
    
    #using official test set

    folder = 'data_train/test_ser_labeled/'

    data_paths_test_official = []

    for file in os.listdir(folder):
        file_path = folder + file
        #to be consistent with rest of the code - we do not want to split these into 4 seconds audios
        data_path = file_path + '_0'
        data_paths_test_official.append(data_path)
        v = file.replace('.wav','').split('_')
        emotion_label = v[len(v)-1]
        data_emotion_target_dictionary[file_path] = emotion_label
    
    random.shuffle(data_paths_test_official)

    ##############################################################################################

    ##############################################################################################

    #run training
    V = 128
    model = make_transformer_encoder_model(V, ri_out_coeffs=2, gen_out_coeffs=2, age_out_coeffs=3,    age_2_out_coeffs=5, N=3, d_model=d_model, d_ff=d_ff, dropout=dropout)
    model_opt = NoamOpt(d_model, 1, 400,
        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    avg_loss=0
    best_val_acc = 0
    best_val_f1_score = 0
    model_path = 'model_test_mfcc.ckpt'
    min_frequency = 0.0
    max_frequency = None

    for epoch in range(20):
        model.train()
        loss, avg_loss, _, _, train_f1_score = run_epoch(model, 
              LossCompute(model.generator, model.ri_generator, model.gen_generator, model.age_generator, model_opt, pretrain),
              data_paths_train, data_emotion_target_dictionary, output_file, training=True, pretrain=pretrain, spec_augment=True, mixup=False, avg_loss=avg_loss, extract_coeffs='mfcc',
              min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=128)
        model.eval()
        with torch.no_grad():
            loss, _, val_acc, val_f1_score, true_val_f1_score = run_epoch(model, 
                    LossCompute(model.generator, model.ri_generator, model.gen_generator, model.age_generator, None, pretrain=pretrain),
                    data_paths_valid, data_emotion_target_dictionary, output_file, training=False, pretrain=pretrain, spec_augment=False, mixup=False, extract_coeffs='mfcc',
                    min_frequency=min_frequency, max_frequency=max_frequency, number_coeffs=128)
            print(val_acc, file=output_file)
            print(true_val_f1_score, file=output_file)


        #if best_val_acc < val_acc:
            #best_val_acc = val_acc
        if best_val_f1_score < true_val_f1_score:
            best_val_f1_score = true_val_f1_score
            print('Saving model', file=output_file)
            torch.save({'model_state_dict': model.state_dict()
            }, model_path)


    #run test
    model_path = 'model_test_mfcc.ckpt'
    checkpoint = torch.load(model_path)
    model = make_transformer_encoder_model(V, ri_out_coeffs=2, gen_out_coeffs=2, age_out_coeffs=3, age_2_out_coeffs=5, N=3, d_model=d_model, d_ff=d_ff, dropout=dropout)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    with torch.no_grad():
        print('End training. Check validation performance', file=output_file)
        print(run_epoch(model, LossCompute(model.generator, model.ri_generator, model.gen_generator, model.age_generator, None, pretrain=pretrain),
                    data_paths_valid, data_emotion_target_dictionary, output_file, training=False, pretrain=pretrain, spec_augment=False, mixup=False, extract_coeffs='mfcc', number_coeffs=128), file=output_file)
        print('End validation. Starting test', file=output_file)
        print(run_epoch(model, LossCompute(model.generator, model.ri_generator, model.gen_generator, model.age_generator, None, pretrain=pretrain),
                    data_paths_test, data_emotion_target_dictionary, output_file, training=False, pretrain=pretrain, spec_augment=False, mixup=False, extract_coeffs='mfcc', number_coeffs=128), file=output_file)
        print('End test. Starting official test', file=output_file)
        print(run_epoch(model, LossCompute(model.generator, model.ri_generator, model.gen_generator, model.age_generator, None, pretrain=pretrain),
                    data_paths_test_official, data_emotion_target_dictionary, output_file, training=False, pretrain=pretrain, spec_augment=False, mixup=False, extract_coeffs='mfcc', number_coeffs=128), file=output_file)


