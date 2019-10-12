import os
import sys

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../')
sys.path.insert(0, root_dir)

from spatial_two_mics.data_generator.my_dregon_dataset_creator_and_storage import create_dataset

# create small but significant dataset
# create_dataset(n_train=256, n_test=128, n_val=128, types_to_mix=['rotor', 'speech'], output_dir='output')

# create a dataset with rotor and speech
# create a dataset with rotor and whitenoise
# create a dataset with whitenoise and speech
r, w, s, c = 'rotor', 'whitenoise', 'speech', 'chirps'
mix_list = [
    [r, s],
    [r, w],
    [w, s]
]
n_train = 832
n_test = 128
n_val = 128
paths = []
for mix_types in mix_list:
    paths.append(create_dataset(n_train, n_test, n_val, types_to_mix=mix_types, output_dir='dataset'))

print('Finished!! data sets created in:')
print(paths)
