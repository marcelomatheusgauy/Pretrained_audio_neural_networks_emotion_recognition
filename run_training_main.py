import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import math, copy, time
from torch.autograd import Variable
import random
import os
import pandas


#load training functions
from run_training import run_training_function

#parameters
d_model_list = [128, 512]
d_ff_list = 4*np.array(d_model_list)
dropout= 0.1
num_repetitions = 25
extract_coeffs = 'mfcc'
pretrain = 'emotion'

#run finetuning function
for d_model_index in range(len(d_model_list)):
    d_model = d_model_list[d_model_index]
    d_ff = d_ff_list[d_model_index]
    for index in range(num_repetitions):
        output_filename = 'Experimental_results/exp_pretrained_emotion_mfcc_' + str(d_model) + '_' + str(d_ff) + '_' + str(index) + '.txt'
        #model_save_path = 'model_pretrained_mfcc_' + str(d_model) + '_' + str(d_ff) + '_' + str(index) + '.ckpt'
        run_training_function(output_filename, extract_coeffs, pretrain, d_model, d_ff, dropout)

