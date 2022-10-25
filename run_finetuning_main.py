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
from run_pretraining import run_pretraining_function
from run_finetuning import run_finetuning_function

#parameters
d_model = 512
d_ff = 4*d_model
dropout = 0.1
num_repetitions = 25
extract_coeffs = 'mfcc'
pretrain = 'emotion'
model_save_path='model_no_coral_pretrain_time_mfcc_1_1.ckpt'

#run finetuning function
model_load_path = model_save_path
for index in range(num_repetitions):
    output_filename = 'Experimental_results/exp_pretrained_emotion_mfcc_' + str(index) + '.txt'
    model_save_path = 'model_test_mfcc.ckpt'
    run_finetuning_function(output_filename, model_load_path, model_save_path, extract_coeffs, pretrain, d_model, d_ff, dropout)


