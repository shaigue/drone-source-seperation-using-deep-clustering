"""!
@brief Dataloader for timit dataset in order to store in an internal
python dictionary structure the whole timit dataset.

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import os
import sys
import scipy.io.wavfile as wavfile
import soundfile as sf
import glob2
import numpy as np
from sphfile import SPHFile

root_dir = os.path.join(
           os.path.dirname(os.path.realpath(__file__)),
           '../../')
sys.path.insert(0, root_dir)

from spatial_two_mics.config import TIMIT_PATH


class TimitLoader(object):
    def __init__(self,
                 normalize_audio_by_std=True):
        self.dataset_path = TIMIT_PATH
        self.normalize_audio_by_std = normalize_audio_by_std

    def wav_converter(self):
        dialects_path = self.dataset_path
        dialects = os.listdir(dialects_path)
        for dialect in dialects:
            dialect_path = os.path.join(dialects_path, dialect)
            speakers = os.listdir(path=dialect_path)
            for speaker in speakers:
                speaker_path = os.path.join(dialect_path, speaker)
                speaker_recordings = os.listdir(path=speaker_path)

                wav_files = glob2.glob(speaker_path + '/*.WAV')

                for wav_file in wav_files:
                    sph = SPHFile(wav_file)
                    txt_file = ""
                    txt_file = wav_file[:-3] + "TXT"

                    f = open(txt_file, 'r')
                    for line in f:
                        words = line.split(" ")
                        start_time = (int(words[0]) / 16000)
                        end_time = (int(words[1]) / 16000)
                    print("writing file ", wav_file)
                    sph.write_wav(wav_file.replace(".WAV", ".wav"), start_time, end_time)

    def get_all_wavs(self, path):
        data_dic = {}
        print("Searching inside: {}...".format(path))
        dialects = os.listdir(path)
        for dial in dialects:
            if dial.startswith('.'):
                continue
            d_path = os.path.join(path, dial)
            speakers = os.listdir(os.path.join(d_path))
            for speaker in speakers:
                if speaker.startswith('.'):
                    continue
                speaker_path = os.path.join(d_path, speaker)
                wavs_paths = glob2.glob(os.path.join(speaker_path,
                                                     '*.wav'))

                # speaker_wavs = []
                # for wav_p in wavs_paths:
                #     with sf.SoundFile(wav_p, mode='r') as f:
                #         speaker_wavs = speaker_wavs + [sf.read(wav_p)] + [wav_p]

                speaker_wavs = [list(sf.read(wav_p)) + [wav_p]
                                for wav_p in wavs_paths]

                if self.normalize_audio_by_std:
                    speaker_wavs = [(sr, wav / np.std(wav), wav_p)
                                    for (wav, sr, wav_p) in speaker_wavs]

                speaker_wavs = [(wav_p.split('/')[-1].split('.wav')[0],
                                {'wav': wav, 'sr': sr, 'path': wav_p})
                                for (sr, wav, wav_p) in speaker_wavs]

                speaker_gender = speaker[0]
                data_dic[speaker] = {
                    'dialect': dial,
                    'gender': speaker_gender,
                    'sentences': dict(speaker_wavs)
                }

        return data_dic

    def load(self):
        """
        Loading all the data inside a dictionary like the one below:

        {
        'train':
            'speaker_id_i': {
                'dialect': which dialect the speaker belongs to,
                'gender': f or m,
                'sentences': {
                    'sentence_id_j': {
                        'wav': wav_on_a_numpy_matrix,
                        'sr': Fs in Hz integer,
                        'path': PAth of the located wav
                    }
                }
            }

        * the same applies for test speakers
        }

        :return: Dictionary
        """
        data_dic = {'train': {},
                    'test': {}
                    }

        for chunk in data_dic:
            wavs_path = os.path.join(self.dataset_path, chunk)
            all_wavs_dic = self.get_all_wavs(wavs_path)
            data_dic[chunk] = all_wavs_dic

        return data_dic


if __name__ == "__main__":
    print("Loading TIMIT Dataset from {}...".format(TIMIT_PATH))
    timit_loader = TimitLoader()
    timit_data = timit_loader.load()