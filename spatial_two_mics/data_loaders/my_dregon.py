"""!
@brief Data loader based on the 'timit.py' code, 
intended for use in my processed DREGON dataset - 'my_dregon'
"""


"""
1. make pre-preprocessing:
    1. make a sample of train / test - for now we will only take the rotor sounds and the clean recordings
    2. process to make it mono-channel, float, and with sample rate of 16kHz
    3. process to make it the same length
    NOTE: this will be implemented in a different file, in 'utils/create_dregon_dataset.py'
2. check if the pre-preprocessing has been made 
    by checking if the directory exists
3. implement the expected dataset loader
"""

import os
import sys
import scipy.io.wavfile as wavfile
import glob2
import numpy as np

root_dir = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           '../../')
sys.path.insert(0, root_dir)

from spatial_two_mics.config import MY_DREGON_PATH2


def check_my_dregon():
    # Checks if the dataset is in the local machine
    return os.path.isdir(MY_DREGON_PATH2)


def get_wav_name(wav_p):
    # fit this to work also in windows
    return os.path.split(wav_p)[1][:-4]
    # return wav_p.split('/')[-1].split('.wav')[0]


def transform_data_dic(data_dic):
    """'<primery_type>':  - for now is either 'clean_sound' or 'rotor_sound'
            {
                '<secondary_type>': subtype inside the primery type,
                {
                    'samples': {
                        '<sample_name>': {
                            'wav': wav_on_a_numpy_matrix,
                            'sr': Fs in Hz integer,
                            'path': Path of the located wav
                        }
                        ... <more sample_name>
                    }
                }
                ... <more secondary_types>
            }
            ... <more primery_types>
            TO:      
            { 
                <type>:
                {
                    <sample_name>:
                    {
                        'wav': [np.array of the sound]
                        'sr': [sample rate]
                        'path': [the path to the audio file]
                    }
                    ... [more samples]
                }
                ... [more names]
            }
    """
    new_dict = {}
    
    for p_type, p_type_dict in data_dic.items():
        for s_type, s_type_dict in p_type_dict.items():
            samples = s_type_dict['samples']
            # for val in samples.values():
            #     val.update({'type': s_type})
            new_dict[s_type] = samples
    
    return new_dict

class MyDregonLoader(object):
    def __init__(self,
                 normalize_audio_by_std=True):
        self.dataset_path = MY_DREGON_PATH2
        self.normalize_audio_by_std = normalize_audio_by_std

    def get_all_wavs(self, path):
        data_dic = {}
        print("Searching inside: {}...".format(path))
        primery_types = os.listdir(path)
        for p_type in primery_types:
            if p_type.startswith('.'):
                continue
            d1_path = os.path.join(path, p_type)
            p_type_dict = {}
            secondary_types = os.listdir(d1_path)
            for s_type in secondary_types:
                if s_type.startswith('.'):
                    continue
                d2_path = os.path.join(d1_path, s_type)
                wavs_paths = glob2.glob(os.path.join(d2_path, '*.wav'))

                wavs = [list(wavfile.read(wav_p)) + [wav_p]
                                for wav_p in wavs_paths]

                if self.normalize_audio_by_std:
                    wavs = [(sr, wav / np.std(wav), wav_p) for (sr, wav, wav_p) in wavs]

                wavs = [(get_wav_name(wav_p),{'wav': wav, 'sr': sr, 'path': wav_p})
                                for (sr, wav, wav_p) in wavs]

                p_type_dict[s_type] = {
                    'samples': dict(wavs)
                }

            data_dic[p_type] = p_type_dict
        
        ## this part is added for simplifing the data representation, 
        ## to only be:
        """'<primery_type>':  - for now is either 'clean_sound' or 'rotor_sound'
            {
                '<secondary_type>': subtype inside the primery type,
                {
                    'samples': {
                        '<sample_name>': {
                            'wav': wav_on_a_numpy_matrix,
                            'sr': Fs in Hz integer,
                            'path': Path of the located wav
                        }
                        ... <more sample_name>
                    }
                }
                ... <more secondary_types>
            }
            ... <more primery_types>
            
            TO:
            
            { 
                <type>:
                {
                    <sample_name>:
                    {
                        'wav': [np.array of the sound]
                        'sr': [sample rate]
                        'path': [the path to the audio file]
                    }
                    ... [more samples]
                }
                ... [more names]
            }

            while type = secondary type. can be either:
                'rotor', 'chirps', 'speech', 'whitenoise'
            """
        data_dic = transform_data_dic(data_dic)

        return data_dic

    def load(self):
        """
        Loading all the data inside a dictionary like the one below:

        {
        'train':
        {
            '<primery_type>':  - for now is either 'clean_sound' or 'rotor_sound'
            {
                '<secondary_type>': subtype inside the primery type,
                {
                    'samples': {
                        '<sample_name>': {
                            'wav': wav_on_a_numpy_matrix,
                            'sr': Fs in Hz integer,
                            'path': Path of the located wav
                        }
                        ... <more sample_name>
                    }
                }
                ... <more secondary_types>
            }
            ... <more primery_types>
        }
        }
        * the same applies for test speakers
        :return: Dictionary
        """
        # NOTE: look at transform datadict for answers
        data_dic = {'train': {},
                    'test': {}
                    }

        for chunk in data_dic:
            wavs_path = os.path.join(self.dataset_path, chunk)
            all_wavs_dic = self.get_all_wavs(wavs_path)
            data_dic[chunk] = all_wavs_dic

        return data_dic



def print_data_dict(data_dict):
    for train_or_test, chunck in data_dict.items():
        print(train_or_test)
        for p_type, p_type_dict in chunck.items():
            print(f'-->primery type: {p_type}')
            for s_type, s_type_dict in p_type_dict.items():
                print(f'---->secondary_type: {s_type}')
                samples = s_type_dict['samples']
                samples_names = list(samples.keys())
                num_samples = len(samples_names)
                print(f'------>number of samples: {num_samples}')
                # print(f'------>names: {samples_names}')



def print_data_dict2(data_dict):
    for train_or_test, chunck in data_dict.items():
        print(train_or_test)
        for s_type, samples in chunck.items():
            print(f'-->type: {s_type}')
            samples_names = list(samples.keys())
            num_samples = len(samples_names)
            print(f'---->number of samples: {num_samples}')


if __name__ == "__main__":
    print("Loading MY_DREGON Dataset from {}...".format(MY_DREGON_PATH2))
    my_dregon_loader = MyDregonLoader()
    my_dregon_data = my_dregon_loader.load()
    ## for testing
    print_data_dict2(my_dregon_data)