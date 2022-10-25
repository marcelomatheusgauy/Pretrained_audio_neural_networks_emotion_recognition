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


#extract data path and index from data path with index

def build_data_paths_with_index(data_paths, data_path):
    data_info = torchaudio.info(data_path)
    total_audio_length = data_info.num_frames
    sample_rate = data_info.sample_rate
    number_audio_files = 1
    if total_audio_length > audio_length*sample_rate:
        number_audio_files = int(math.ceil((total_audio_length/sample_rate)-audio_length+1))
    for index in range(number_audio_files):
        data_path_with_index = data_path + '_' + str(index)
        data_paths.append(data_path_with_index)


def extract_index_from_path(data_path_with_index):
    if (data_path_with_index.find('wav') != -1):
        _position_in_path = data_path_with_index.find('wav')+3
    elif (data_path_with_index.find('mp3') != -1):
        _position_in_path = data_path_with_index.find('mp3')+3
    else:
        _position_in_path = data_path_with_index.find('ogg')+3
    data_path = data_path_with_index[:_position_in_path]
    index = int(data_path_with_index[_position_in_path+1:])
    return data_path, index

class Mixup(object):
    def __init__(self, mixup_alpha):
        """Mixup coefficient generator.
        """
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState()

    def get_lambda(self, batch_size):
        """Get mixup random coefficients.
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        """
        mixup_lambdas = []
        for n in range(batch_size):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]
            mixup_lambdas.append(lam)
            #mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)

#build function to process data in batches
def process_batches(data_paths, data_emotion_target_dictionary, number_coeffs, extract_coeffs, min_frequency, max_frequency, batch_size, pretrain, spec_augment, mixup, path_index):
    
    #parameters below maybe should be defined elsewhere
    #set audio length in seconds - this is max length of audios
    audio_length = 16
    device = 'cuda'
    new_sample_rate = 8000
        
    ################################
    
    data_batch = []
    
    data_emotion_target_list = []
    
    while len(data_batch) < batch_size and path_index < len(data_paths):
        data_path_with_index = data_paths[path_index]
        data_path, index = extract_index_from_path(data_path_with_index)
        sample_rate = torchaudio.info(data_path).sample_rate
        
        data_elem, sample_rate = torchaudio.load(data_path, frame_offset=index*sample_rate, num_frames = audio_length*sample_rate)
        #downsampling to fit gpu memory
        data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
        sample_rate = new_sample_rate
        data_elem = data_elem[0]
        
        data_batch.append(data_elem)
        
        #for supervised training we store data about the file
        #respiratory insufficiency
        if pretrain == 'emotion':
            if data_emotion_target_dictionary[data_path] == 'neutral':
                data_emotion_target_list.append(0)
            elif data_emotion_target_dictionary[data_path] == 'non-neutral-male':
                data_emotion_target_list.append(1)
            elif data_emotion_target_dictionary[data_path] == 'non-neutral-female':
                data_emotion_target_list.append(2)
            else:#this should not happen
                data_emotion_target_list.append(3)
                    
        path_index +=1

    if mixup==True:
        for batch_index in range(batch_size):
            random_pair_index = np.random.randint(len(data_paths))
            data_path_with_index = data_paths[random_pair_index]
            data_path, index = extract_index_from_path(data_path_with_index)
            sample_rate = torchaudio.info(data_path).sample_rate

            data_elem, sample_rate = torchaudio.load(data_path, frame_offset=index*sample_rate, num_frames = audio_length*sample_rate)
            #downsampling to fit gpu memory
            data_elem = torchaudio.transforms.Resample(sample_rate, new_sample_rate)(data_elem)
            sample_rate = new_sample_rate
            data_elem = data_elem[0]
            
            data_batch.append(data_elem)
            
            if pretrain == 'emotion':
                if data_emotion_target_dictionary[data_path] == 'neutral':
                    data_emotion_target_list.append(0)
                elif data_emotion_target_dictionary[data_path] == 'non-neutral-male':
                    data_emotion_target_list.append(1)
                elif data_emotion_target_dictionary[data_path] == 'non-neutral-female':
                    data_emotion_target_list.append(2)
                else:#this should not happen
                    data_emotion_target_list.append(3)
            
        
    #convert list to torch tensor (pads different audio lengths to same size)
    data_batch = nn.utils.rnn.pad_sequence(data_batch, batch_first=True)
    
    data_batch = data_batch.to(device)
    
    #for supervised training
    data_emotion_target_list = torch.LongTensor(data_emotion_target_list)
    data_emotion_target_list = data_emotion_target_list.to(device)
    ###########################
    
    if extract_coeffs == 'mfcc':#extract MFCC from data
        data_batch = torchaudio.transforms.MFCC(sample_rate, number_coeffs, melkwargs={"f_min":min_frequency, "f_max":max_frequency}).to(device)(data_batch)
    elif extract_coeffs == 'mel':#extract MelSpectrogram otherwise
        data_batch = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=number_coeffs, f_min=min_frequency, f_max=max_frequency).to(device)(data_batch)
    else:
        data_batch_1 = torchaudio.transforms.MFCC(sample_rate, number_coeffs, melkwargs={"f_min":min_frequency, "f_max":max_frequency}).to(device)(data_batch)
        data_batch_2 = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=number_coeffs, f_min=min_frequency, f_max=max_frequency).to(device)(data_batch)
        data_batch = torch.cat([data_batch_1, data_batch_2], dim = 1)

    
        mixup_data_batch = nn.utils.rnn.pad_sequence(mixup_data_batch, batch_first=True)
        mixup_data_batch = mixup_data_batch.to(device)
        mixup_data_emotion_target_list = torch.LongTensor(mixup_data_emotion_target_list)
        mixup_data_emotion_target_list = mixup_data_emotion_target_list.to(device)
        
        if extract_coeffs == 'mfcc':#extract MFCC from data
            mixup_data_batch = torchaudio.transforms.MFCC(sample_rate, number_coeffs, melkwargs={"f_min":min_frequency, "f_max":max_frequency}).to(device)(mixup_data_batch)
        elif extract_coeffs == 'mel':#extract MelSpectrogram otherwise
            mixup_data_batch = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=number_coeffs, f_min=min_frequency, f_max=max_frequency).to(device)(mixup_data_batch)
        else:
            mixup_data_batch_1 = torchaudio.transforms.MFCC(sample_rate, number_coeffs, melkwargs={"f_min":min_frequency, "f_max":max_frequency}).to(device)(mixup_data_batch)
            mixup_data_batch_2 = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=number_coeffs, f_min=min_frequency, f_max=max_frequency).to(device)(mixup_data_batch)
            mixup_data_batch = torch.cat([mixup_data_batch_1, mixup_data_batch_2], dim = 1)
            
            
        mixup_augmenter = Mixup(mixup_alpha = 1.)
        mixup_lambda = mixup_augmenter.get_lambda(batch_size)
        
        #convert to numpy
        data_batch = data_batch.cpu().numpy()
        data_emotion_target_list = data_emotion_target_list.cpu().numpy()

        data_batch = data_batch[:batch_size] * mixup_lambda + data_batch[batch_size:] * (1-mixup_lambda)
        data_emotion_target_list = data_emotion_target_list[:batch_size] * mixup_lambda + data_emotion_target_list[batch_size:] * (1-mixup_lambda)
        
        #convert back
        data_batch = torch.LongTensor(data_batch)
        data_batch = data_batch.to(device)
        data_emotion_target_list = torch.LongTensor(data_emotion_target_list)
        data_emotion_target_list = data_emotion_target_list.to(device)

        
    #blocks size - we block multiple audio frames into one
    block_size = 1
    time_steps = data_batch.shape[2]
    time_blocks = int(math.floor(time_steps/block_size))
    data_batch = data_batch[:,:,:block_size*time_blocks]
    data_batch = torch.reshape(data_batch, (data_batch.shape[0], block_size*data_batch.shape[1], time_blocks))
    
    #permute so we have batch_size, time, n_coeffs
    data_batch = data_batch.permute(0,2,1)
    #print(data_batch.shape)
    
    data_batch_masked = copy.deepcopy(data_batch)
    
    #for emotion recognition we do specaugment and mixup during training
    if pretrain == 'emotion' and spec_augment == True:
        #specaugment
        batch_size = data_batch.shape[0]
        time_len = data_batch.shape[1]
        n_coeffs = data_batch.shape[2]
        
        for idx in range(batch_size):
            #time masking
            perm_length = np.random.randint(64)
            valid_start_max = max(time_len - perm_length - 1, 0)
            chosen_starts = torch.randperm(valid_start_max + 1)[:1]
            chosen_intervals = _starts_to_intervals(chosen_starts, perm_length)
            data_batch_masked[idx, chosen_intervals, :] = 0
            
            #first frequency masking
            perm_length = np.random.randint(8)
            valid_start_max = max(n_coeffs - perm_length - 1, 0)
            chosen_starts = torch.randperm(valid_start_max + 1)[:1]
            chosen_intervals = _starts_to_intervals(chosen_starts, perm_length)
            data_batch_masked[idx, :, chosen_intervals] = 0
            #second frequency masking
            perm_length = np.random.randint(8)
            valid_start_max = max(n_coeffs - perm_length - 1, 0)
            chosen_starts = torch.randperm(valid_start_max + 1)[:1]
            chosen_intervals = _starts_to_intervals(chosen_starts, perm_length)
            data_batch_masked[idx, :, chosen_intervals] = 0
            

    
    return data_batch_masked, data_batch, data_emotion_target_list, path_index

#this function is used to mask along multiple consecutive frames - see https://github.com/s3prl/s3prl/blob/master/pretrain/mockingjay/task.py
def _starts_to_intervals(starts, consecutive):
    tiled = starts.expand(consecutive, starts.size(0)).permute(1, 0)
    offset = torch.arange(consecutive).expand_as(tiled)
    intervals = tiled + offset
    return intervals.view(-1)


#function to train model
def run_epoch(model, loss_compute, data_paths, data_emotion_target_dictionary, output_file, training=True, avg_loss=0, pretrain='pretrain', spec_augment=True, mixup=True, batch_size=16, extract_coeffs='both', min_frequency = 0.0, max_frequency=None, number_coeffs=128):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    train_acc_avg = 0
    f1_score_avg = 0
    
    #compute f1_score
    true_f1_score = 0
    outputs = []
    targets = []
    
    number_elements = len(data_paths)
    #number_steps = int(math.ceil(number_elements/batch_size))
    
    #path index is the index of the audio file in the filenames list
    path_index = 0
    #pos_index is the index at which we extract a part of the audio file (if it is too long - otherwise pos_index is 0)
    pos_index = 0
    #step index stores the amount of steps taken by the algorithm so far
    step_index = 0
    while path_index < number_elements:
        step_index +=1
        #load the data and mask it
        data_batch_masked, data_batch, data_emotion_target_list, path_index = process_batches(data_paths, data_emotion_target_dictionary, number_coeffs, extract_coeffs, min_frequency, max_frequency, batch_size, pretrain, spec_augment, mixup, path_index)
        b_size = data_batch_masked.shape[0]

        src_mask = (data_batch[:,:,0] != 0).unsqueeze(1)
        #pass data through transformer
        out = model.forward(data_batch_masked, src_mask)
        

        loss, train_acc, f1_score, target, output = loss_compute(out, data_emotion_target_list, training)
        
        outputs.append(output)
        targets.append(target)
        
        total_loss += loss
        avg_loss = avg_loss*0.99 + loss*0.01
        train_acc_avg = (train_acc_avg*(step_index-1)+train_acc)/(step_index)
        f1_score_avg = (f1_score_avg*(step_index-1)+f1_score)/(step_index)
        total_tokens += b_size
        tokens += b_size
        
        #if path_index > 10:
        #    break
        
        if step_index % 5 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f Train_acc: %f F1_score: %f" %
                    (step_index, avg_loss, tokens / elapsed, train_acc_avg, f1_score_avg), file=output_file)
            start = time.time()
            tokens = 0
            
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    true_f1_score = sklearn.metrics.f1_score(outputs, targets, labels=[0,1,2], average='macro')
    print('Final F1_score=', true_f1_score, file=output_file)
    
    return total_loss / (total_tokens), avg_loss, train_acc_avg, f1_score_avg, true_f1_score

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

class LossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, ri_generator, gen_generator, age_generator, opt=None, pretrain='pretrain'):
        self.generator = generator
        self.ri_generator = ri_generator
        self.gen_generator = gen_generator
        self.age_generator = age_generator
        self.opt = opt
        self.pretrain = pretrain
        
    def __call__(self, x, y, training):
        train_acc = 0
        f1_score=0
        x = self.age_generator(x)

        cross_entropy_loss = nn.CrossEntropyLoss()

        loss = cross_entropy_loss(x[:,0], y)
        _, predicted = torch.max(x[:,0:1], 2)
        #print('pred=', predicted.shape)
        train_acc = torch.sum(predicted[:,0]==y)/y.shape[0]
        preds = predicted[:,0].detach().cpu().clone()
        y_true = y.detach().cpu().clone()
        f1_score = sklearn.metrics.f1_score(y_true, preds, labels=[0,1,2], average='macro')
        
        if training == True:
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
            #print(loss.data.shape)
        return loss.data.item(), train_acc, f1_score, y_true, preds
    


