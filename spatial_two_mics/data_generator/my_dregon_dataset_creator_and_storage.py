"""
@breif  This is the code for creating the fast data set for the MY_DREGON dataset
        it creates the mixtures with random positionings, the tf domain representation
        and the masks for the model running, to save time in the model trining.

NOTE: This script will not use argument parsing, just be run through a differnt python script
@author - shaigue
"""
# ______________________________IMPORTS__________________________________________________
import argparse
import os
import sys
import numpy as np
from random import shuffle, randint
from pprint import pprint
import joblib

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.data_loaders.my_dregon as my_dregon_loader
import spatial_two_mics.data_generator.source_position_generator as \
    positions_generator
import spatial_two_mics.labels_inference.tf_label_estimator as \
    mask_estimator
import spatial_two_mics.utils.audio_mixture_constructor as \
            mix_constructor
import spatial_two_mics.utils.progress_display as progress_display
# ______________________________________________________________________________________


#_____________________________HELPER FUNCTIONS__________________________________________
def load_data():
    """
    Uses the my_dregon data loader implementation to load the data
    and returns the data_dict the load() function produces
    current implementation returns a dictionary that looks like this:

    {
        'train':
        { 
            <sound_type>:
            {
                <sample_name>:
                {
                    'wav': [np.array of the sound]
                    'sr': [sample rate]
                    'path': [the path to the audio file]
                }
                ... [more sample_names]
            }
            ... [more sound_types]
        }
        'test' - ***the same as 'train'***
    } 
    NOTE: change this in order to change the dataset used
    """
    data_loader = my_dregon_loader.MyDregonLoader(normalize_audio_by_std=True)
    return data_loader.load()


def validation_split(data_dict: dict, 
    split_train: bool = True, 
    split_ratio: float = 0.2) -> dict: 
    """Creates a new section for validation.
    
    Args:
        data_dict: The given dictionary to split a part of for validation
        split_train: True if to take from the train, False for test
        split_ratio: The ratio of the samples to take for validation

    Returns: The same samples, but with a new 'val' partition 
    """

    split_target = 'test'
    if split_train:
        split_target = 'train'
    split_section = data_dict[split_target]
    
    val_dict = {}
    for sound_type, samples in split_section.items():
        # for each type, split the samples to validation
        num_total_samples = len(samples)
        num_val_samples = int(num_total_samples * split_ratio)
        val_dict[sound_type] = {}

        keys_to_move = [k for k in samples.keys()]
        keys_to_move = keys_to_move[:num_val_samples]

        for k in keys_to_move:
            val_dict[sound_type][k] = samples.pop(k)
    
    data_dict['val'] = val_dict

    return data_dict


def partition_iter(n_train: int, n_test: int, n_val: int) -> list:
    return [('train', n_train), ('test', n_test), ('val', n_val)]


def calc_max_mixtures(partition_dict: dict, types_to_mix: list):
    n = 1
    for sound_type in types_to_mix:
        k = len(partition_dict[sound_type])
        ## debug
        ## print(f"{k} len of {sound_type}")
        ##
        n *= k
    return n


def check_n_mixture(partition_dict: dict,
    types_to_mix: list,
    n_mixtures: int,
    partition: str) -> None:
    max_mixtures = calc_max_mixtures(partition_dict, types_to_mix)
    if max_mixtures < n_mixtures:
        raise ValueError(f'''number of requested mixtures is not possible.
                            requested: {n_mixtures}, max: {max_mixtures}, for {partition}''')

def random_indices_generator(ranges: list) -> list:
    """Generates unique vector indices."""

    combinations_used = set()
    max_num_combinations = 1
    for max_i in ranges:
        max_num_combinations *= max_i
    
    combinations_created = 0
    while combinations_created < max_num_combinations:
        combination = [randint(0, max_i - 1) for max_i in ranges]
        combination_str = '-'.join(map(str, combination))
        if combination_str in combinations_used:
            continue
        
        combinations_used.add(combination_str)
        combinations_created += 1
        yield combination


def create_sources_ids(types_to_mix: list, 
    samples_by_type: list, 
    indices: list, 
    partition_dict: dict) -> dict:
    """Creates a mixture information for a set of indices."""
    
    sources_ids = []
    for i, sound_type in enumerate(types_to_mix):
        sample_id = samples_by_type[i][indices[i]]
        wav_path = partition_dict[sound_type][sample_id]['path']
        source_id = {'sound_type': sound_type, 'sample_id': sample_id, 'wav_path': wav_path}
        sources_ids.append(source_id)

    return sources_ids


def create_mixtures(n_mixtures: int, 
    types_to_mix: list, 
    partition_dict: dict) -> list:
    """Mixes different sources from the current partition to make unique mixtures.

    Takes different elements from the samples, and mixes them to create the mixtures,
    in a unique way.

    Args:
        n_mixtures: The number of mixtures to produce.
        types_to_mix: A list of the sound types that we will mix.
            The length of the list also determins the mixture length.
        partition_dict: A dictionary 2 layers of keys, the first layer is the sound type,
            the second layer is the sample name, and inside each entry a dictionary of the form
            {'wav': np.array, 'sr': int, 'wav_path: str}

    Returns:
        A list of the size n_mixtures, where each entry is a unique combination of
        samples from each sound type, the same order as types_to_mix. In the form of:
        {'sound_type': str, 'sample_id': str, 'wav_path': str}(for each item on the list)
    """
    
    # generate a list of all of the samples in each sound type, and mix them
    samples_by_type = []
    for sound_type in types_to_mix:
        samples_ids = [k for k in partition_dict[sound_type].keys()]
        samples_by_type.append(samples_ids)
    
    # generate a unique set of indices for each mixture for the lists:
    ranges = [len(l) for l in samples_by_type]
    indices_generator = random_indices_generator(ranges)

    # create for each unique indices vector a valid combination
    mixtures = []
    for i in range(n_mixtures):
        indices = next(indices_generator)
        sources_ids = create_sources_ids(types_to_mix, samples_by_type, indices, partition_dict)
        mixtures.append(sources_ids)
    
    return mixtures


class MixtureDataCreator(object):
    """A function object that recieves the number of sources, \
        and generates for a given mixture of the correct length \
        the required mixture data.

        Args:
            n_sources: the number of different sources to be mixed

        Returns:
            The data dict for the given mixture, structured like that:
            {
                'ground_truth_mask': np.array,
                'soft_labeled_mask': np.array,
                'row_phase_diff': np.array,
                'real_tfs': np.array,
                'imag_tfs': np.array,
                'wavs': np.array,
            }
    """
    def __init__(self, n_sources: int):
        # check that n_sources is positive
        if n_sources < 1:
            raise ValueError('n_sources should be positive integer')

        self.n_sources = n_sources
        self.positions_generator = positions_generator.RandomCirclePositioner()
        self.mixture_generator = mix_constructor.AudioMixtureConstructor(n_fft=512,
            win_len=512, hop_len=128, mixture_duration=1.5, force_delays=None)
        self.gt_estimator = mask_estimator.TFMaskEstimator(
                             inference_method='Ground_truth')
        self.sl_estimator = mask_estimator.TFMaskEstimator(
                             inference_method='duet_Kmeans',
                             return_duet_raw_features=True)
        

    def __call__(self, mixture: list) -> dict:
        if len(mixture) != self.n_sources:
            raise ValueError('The length of the mixture should match n_sources')

        positions = self.positions_generator.get_sources_locations(self.n_sources)
        mixture_info = {'positions': positions, 'sources_ids': mixture}
        tf_mixture = self.mixture_generator.construct_mixture(mixture_info)
        
        gt_mask = self.gt_estimator.infer_mixture_labels(tf_mixture)
        duet_mask, raw_phase_diff = self.sl_estimator.infer_mixture_labels(tf_mixture)
        normalized_raw_phase = np.clip(raw_phase_diff, -2., 2.)
        normalized_raw_phase -= normalized_raw_phase.mean()
        normalized_raw_phase /= normalized_raw_phase.std() + 10e-12
        
        data = {}
        data['soft_labeled_mask'] = duet_mask
        data['raw_phase_diff'] = np.asarray(normalized_raw_phase, dtype=np.float32)
        data['ground_truth_mask'] = gt_mask
        data['real_tfs'] = np.real(tf_mixture['m1_tf'])
        data['imag_tfs'] = np.imag(tf_mixture['m1_tf'])
        data['wavs'] = tf_mixture['sources_raw']
        
        return data


def generate_save_path(output_dir: str, partition: str, i: int) -> str:
    """Generates a path to save the spacific mixture."""
    return os.path.join(output_dir, partition, str(i))


def save_mixture_data(mixture_data: dict, mixture_save_path: str) -> None:
    """Saves the mixture data into files."""
    if not os.path.isdir(mixture_save_path):
        os.makedirs(mixture_save_path)
    
    for k, v in mixture_data.items():
        file_path = os.path.join(mixture_save_path, k)
        joblib.dump(v, file_path, compress=0)


def add_dataset_subdir(output_dir: str, 
    n_train: int, n_test: int, n_val: int, 
    types_to_mix: list) -> str:
    """Returns a unique dataset dir path, using the creation parameters."""

    name = '_'.join(types_to_mix) + '-' + '-'.join(map(str, [n_train, n_test, n_val]))
    return os.path.join(output_dir, name)


def print_progress(message: str, n_stars: int = 0) -> None:
    pre = '*' * n_stars
    print(f'\n{pre}{message}\n')


#_______________________________________________________________________________________

#_____________________________DATA CREATION_____________________________________________
def create_dataset(n_train: int, n_test: int, n_val:int, 
    types_to_mix: list,   # a list of n [n >= 2] types of sounds, string to have in the mixtures
                        # the number of sources to mix will be infered from the length of this list
    output_dir: str) -> str:
    """Creates the dataset.

    This function creates mixtures of 2 simulated microphones and spatial fetures
    it cretes the time-frequancy representatins of the mixed samples, 
    and saves them into the output directory. 
    Creates train, test, and validation sets.

    Args:
        n_train: Number of train mixtures
        n_test: Number of test mixtures
        n_val: Number of validation mixtures
        types_to_mix: A list of n [n >= 2] types of sounds, to have in the mixtures.
            The number of sources to mix will be infered from the length of this list.
        output_dir: The output directory of all the datasets to train on.
            The created dataset will be saved in a sub-directory in there, 
            with a name corresponding to the parametrs given.

    Returns: The path to the root of the created dataset
    """

    """
    the file structure of the folders after it should look like:

    output_dir
    +-- test
    |   +-- <mixture_name_0>
    |       +-- real_tfs
    |       +-- imag_tfs
    |       +-- abs_tfs
    |       +-- ground_truth_mask
    |       +-- soft_labeled_mask
    |       +-- wavs
    |       +-- raw_phase_diff
    |   +-- <mixture_name_1>
    |   ***The same as 0***
    ...
    |   +-- <mixture_name_(n_test - 1)>
    |   ***The same as 0***
    +-- val
    ***The same as test***
    +-- train
    ***The same as test***
    """
    n_sources = len(types_to_mix)
    output_dir = add_dataset_subdir(output_dir, n_train, n_test, n_val, types_to_mix)

    print_progress(f'creating the dataset in {output_dir}')
    print_progress('LOADING DATA')
    # load the data using the data loader
    data_dict = load_data()
    # split the training data to 'train' 'val' subsets
    data_dict = validation_split(data_dict)
 
    mixture_data_creator = MixtureDataCreator(n_sources)

    # repeat for the patrition 'train', 'val', 'test' -
    for partition, n_mixtures in partition_iter(n_train, n_test, n_val):
        print_progress(f'Creating partition: {partition} with {n_mixtures} mixtures.', 1)

        partition_dict = data_dict[partition]
        # check if the required number of mixtures('n_mixtures') can be created
        check_n_mixture(partition_dict, types_to_mix, n_mixtures, partition)
        
        # make 'n_mixtures' unique combinations of sample of the appropriate types
        mixtures = create_mixtures(n_mixtures, types_to_mix, partition_dict)
        for i, mixture in enumerate(mixtures):
            if i % 10 == 0:
                print_progress(f'{i} out of {n_mixtures} created', 2)
            # produce the required data to store
            mixture_data = mixture_data_creator(mixture)
            # generate the path to save the samples in, and save the samples in there
            mixture_save_path = generate_save_path(output_dir, partition, i)
            save_mixture_data(mixture_data, mixture_save_path)

    # data generation is complete
    return output_dir

#_______________________________________________________________________________________


#_____________________________TESTING___________________________________________________
if __name__ == '__main__':
    # test data loading
    # data_dict = load_data()
    # data_dict = validation_split(data_dict)
    # my_dregon_loader.print_data_dict2(data_dict)

    # partition_dict = data_dict['val']
    
    # should pass first fail second:
    # check_n_mixture(partition_dict, ['whitenoise', 'rotor'], 10, 'val')
    # check_n_mixture(partition_dict, ['whitenoise', 'rotor'], 100000, 'val')

    # test: random_indices_generator()
    # ranges = [3, 5, 2]
    # s = 0
    # for I in random_indices_generator(ranges):
    #     s += 1
    #     print(I)
    # print(s)

    # test: create_sources_ids()
    # types_to_mix = ['whitenoise', 'speech', 'rotor']
    # samples_by_type = []
    # for t in types_to_mix:
    #     samples_by_type.append([k for k in partition_dict[t].keys()])
    # indices = [1, 1, 1]
    # sources_ids = create_sources_ids(types_to_mix, samples_by_type, indices, partition_dict)
    # print([samples_by_type[i][indices[i]] for i in range(3)])
    # print(sources_ids)

    # test: create_mixtures()
    # n_mixtures = 10
    # mixtures = create_mixtures(n_mixtures, types_to_mix, partition_dict)
    # pprint(mixtures)

    # random_positioner = positions_generator.RandomCirclePositioner()
    # positions = random_positioner.get_sources_locations(3)
    # pprint(positions)
    
    # test = create_dataset(10, 10, 10, ['rotor', 'speech'], 'output', None)
    # pprint(test)

    # print(generate_save_path('shai', 'train', 7))
    create_dataset(20, 10, 10, ['rotor', 'speech'], 'output')

    print('FINISHED')    
#_______________________________________________________________________________________