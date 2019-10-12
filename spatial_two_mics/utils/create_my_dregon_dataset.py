"""
1. make pre-preprocessing:
    1. make a sample of train / test - for now we will only take the rotor sounds and the clean recordings
    2. process to make it mono-channel, float, and with sample rate of 16kHz
    3. process to make it the same length
"""

import os
import sys
import scipy.io.wavfile as wavfile
import librosa
import glob2
import numpy as np
import random
import soundfile as sf

root_dir = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           '../../')
sys.path.insert(0, root_dir)

from spatial_two_mics.config import DREGON_PATH, MY_DREGON_PATH, MY_DREGON_PATH2

import shutil

"""
The file structure should look like this:

/
----clean_sound
    ----***diferent types of noises, directories***
----flight_nosource
    ----***different types of flight patterns, directories***
----flight_with_source
    ----***different types of sounds and flight, directories***
----rotor_sound
    ----***just a single directory with the individual rotor sound***

The target shoud be:

/
----train
    ----clean_sound
        ----***different sounds directories***
    ----rotor_sound
        ----***different sounds directories***
----test
    ***the same as train, just fewer samples***

"""

def check_dregon_dataset_structure(dregon_path, dirs_to_process):
    # make sure that the directory exists
    if not os.path.isdir(dregon_path):
        return False
    # make sure that the required sub-directories exists
    dirs_path = [os.path.join(dregon_path, d) for d in dirs_to_process]
    for d in dirs_path:
        if not os.path.isdir(d):
            return False
    # make sure that the first directory inside has at least one .wav file
    for d in dirs_path:
        # print('***', d, '***')
        d = glob2.glob(os.path.join(d, '*'))[0]
        wav_files = glob2.glob(os.path.join(d, "*.wav"))
        if len(wav_files) == 0:
            return False

    return True


def concatenate_channels(multichannel_data, n_channels):
    concatenated_data = np.concatenate(multichannel_data[:, :n_channels])
    return concatenated_data


def process_wav_file(wav, orig_d_path, target_d_path, sample_duration, sample_rate, channels_to_extract):
    # read the orig wav file, and resample with given sample_rate
    wav_p = os.path.join(orig_d_path, wav)
    data, sr = sf.read(wav_p, dtype=np.float32)
    if data.shape[1] < channels_to_extract:
        raise ValueError('Not enough channels')
    # reduce it to a single channel and resample
    data = np.concatenate(data[:, :channels_to_extract])
    data = librosa.resample(data, sr, sample_rate)
    # split it into the required duration
    frames_per_sample = int(sample_duration * sample_rate)
    # write the audio to the given directory
    idx = 0
    start = 0
    end = frames_per_sample
    while end < data.shape[0]:
        target_wav = wav[:-4] + '_' + str(idx) + '.wav'
        target_wav_p = os.path.join(target_d_path, target_wav)
        wavfile.write(target_wav_p, sample_rate, data[start:end])
        
        idx += 1
        start = end
        end += frames_per_sample


def get_all_wav_files(dir_path):
    wav_files = os.listdir(dir_path)
    return list(filter(lambda x: x.split('.')[-1] == 'wav', wav_files))


def process_sub_dir(sub_dir, dregon_path, target_path, sample_duration, sample_rate, channels_to_extract):
    target_dir_path = os.path.join(target_path, sub_dir)
    orig_dir_path = os.path.join(dregon_path, sub_dir)
    print(f'***processing {orig_dir_path}***')
    # create the sub-directory in the target
    os.mkdir(target_dir_path)

    # start reading the directories inside of the original path
    for d in os.listdir(orig_dir_path):
        orig_d_path = os.path.join(orig_dir_path, d)
        target_d_path = os.path.join(target_dir_path, d)

        print(f'***->processing {orig_d_path}***')
        
        # create the directory in the path
        os.mkdir(target_d_path)
        
        # take all of the wav files in the orig directory
        wav_files = get_all_wav_files(orig_d_path)
        ## print(f'***->list of files: {wav_files}')
        # process each wav file seperatly
        for wav in wav_files:
            process_wav_file(wav, orig_d_path, target_d_path, sample_duration, sample_rate, channels_to_extract)
            
def split_train_test(target_path, test_ratio):
    """
    walks in every sub directory, 2 layers down, and for every directory splits it into 
    train and test
    """
    # start by creating the test/train directories:
    test_p = os.path.join(target_path, 'test')
    train_p = os.path.join(target_path, 'train')
    os.mkdir(test_p)
    os.mkdir(train_p)

    # walk the first layer
    for d1 in os.listdir(target_path):
        if d1 == 'test' or d1 == 'train':
            continue
        d1_p = os.path.join(target_path, d1)
        if not os.path.isdir(d1_p):
            continue
        d1_test_p = os.path.join(test_p, d1)
        d1_train_p = os.path.join(train_p, d1)
        os.mkdir(d1_test_p)
        os.mkdir(d1_train_p)
        # walk the second layer
        for d2 in os.listdir(d1_p):
            d2_p = os.path.join(d1_p, d2)
            if not os.path.isdir(d2_p):
                continue
            d2_test_p = os.path.join(d1_test_p, d2)
            d2_train_p = os.path.join(d1_train_p, d2)
            os.mkdir(d2_test_p)
            os.mkdir(d2_train_p)
            # collect all the wav files and split them randomly
            wav_files = get_all_wav_files(d2_p)
            random.shuffle(wav_files)
            num_test = int(test_ratio * len(wav_files))
            print(f'moving files from: {d2_p}')
            ## print(f'the files are: {wav_files}')
            print(f'test: {num_test}, train: {len(wav_files) - num_test}')
            # move the first files to test and the last to train
            for f in wav_files[:num_test]:
                orig_f = os.path.join(d2_p, f)
                target_f = os.path.join(d2_test_p, f)
                shutil.move(orig_f, target_f) 
            for f in wav_files[num_test:]:
                orig_f = os.path.join(d2_p, f)
                target_f = os.path.join(d2_train_p, f)
                shutil.move(orig_f, target_f)
            
            # delete the folders
            os.rmdir(d2_p)
        os.rmdir(d1_p)

def main(args):
    """The main function of the module, 
    does the dataset creation from the original DREGON detaset"""
    # open the args
    test_ratio = args['test_ratio']
    dregon_path = args['dregon_path']
    target_path = args['target_path']
    dirs_to_process = args['dirs_to_process']
    sample_rate = args['sample_rate']
    sample_duration = args['sample_duration']
    channels_to_extract = args['channels_to_extract']

    # check if the path has the right structure
    # if not let the user know
    if not check_dregon_dataset_structure(dregon_path, dirs_to_process):
        raise IOError('Something in the stracture of the given DREGON directory is not good')
 
    # create the main directory for the new dataset
    # if it already exists, then erase it and make a new one
    if os.path.isdir(target_path):
        shutil.rmtree(target_path)
    
    os.mkdir(target_path)
    
    # sub-dir processing processing
    for sub_dir in dirs_to_process:
        process_sub_dir(sub_dir, dregon_path, target_path, sample_duration, sample_rate,
            channels_to_extract)

    # now split it into test train
    split_train_test(target_path, test_ratio)

    # end with success
    return 0

if __name__ == '__main__':
    # TODO: add parser
    args =  {
        'test_ratio': 0.20, # the ratio of the samples that goes twords testing
        # a list of the sub directories to process
        'dirs_to_process': ['clean_sound', 'rotor_sound'],
        'dregon_path': DREGON_PATH,
        'target_path': MY_DREGON_PATH2,
        'sample_duration': 2, # the duration of one sample, in seconds
        'sample_rate': 16000, # 16kHz
        'channels_to_extract': 3 # take every channel as a source
    }
    main(args)
    # shutil.rmtree(os.path.join(MY_DREGON_PATH, 'test'))
    # shutil.rmtree(os.path.join(MY_DREGON_PATH, 'train'))
    # split_train_test(args['target_path'], args['test_ratio'])
