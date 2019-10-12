"""!
@brief A simple experiment on how models, losses, etc should be used

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of illinois at Urbana Champaign
"""

import argparse
import os
import sys
import torch
import time
import numpy as np
import copy
from pprint import pprint
from torch.utils.data import DataLoader

root_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '../../../')
sys.path.insert(0, root_dir)

import spatial_two_mics.dnn.models.simple_LSTM_encoder as LSTM_enc
import spatial_two_mics.dnn.losses.affinity_approximation as \
    affinity_losses
import spatial_two_mics.dnn.utils.dataset as data_generator
import spatial_two_mics.dnn.utils.data_conversions as converters


def check_device_model_loading(model):
    device = 0
    print(torch.cuda.get_device_capability(device=device))
    print(torch.cuda.memory_allocated(device=device))
    print(torch.cuda.memory_cached(device=device))

    model = model.cuda()
    print(torch.cuda.get_device_properties(device=device).total_memory)
    print(torch.cuda.memory_allocated(device))
    print(torch.cuda.memory_cached(device))

    temp_model = copy.deepcopy(model)
    temp_model = temp_model.cuda()
    print(torch.cuda.max_memory_cached(device=device))
    print(torch.cuda.memory_allocated(device))
    print(torch.cuda.memory_cached(device))


def compare_losses(vs, one_hot_ys):
    timing_dic = {}

    before = time.time()
    flatened_ys = one_hot_ys.view(one_hot_ys.size(0),
                                  -1,
                                  one_hot_ys.size(-1)).cuda()
    naive_loss = affinity_losses.naive(vs, flatened_ys)
    now = time.time()
    timing_dic['Naive Loss Implementation'] = now - before

    before = time.time()
    expanded_vs = vs.view(vs.size(0), one_hot_ys.size(1),
                          one_hot_ys.size(2), vs.size(-1)).cuda()
    diagonal_loss = affinity_losses.diagonal(expanded_vs,
                                             one_hot_ys)
    now = time.time()
    timing_dic['Diagonal Loss Implementation'] = now - before

    pprint(timing_dic)

    return diagonal_loss


def example_of_usage(args):

    visible_cuda_ids = ','.join(map(str, args.cuda_available_devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = visible_cuda_ids
    print(visible_cuda_ids)
    print(torch.cuda.current_device())

    training_generator, n_batches = data_generator.get_data_generator(
        args)
    timing_dic = {}

    before = time.time()
    model = LSTM_enc.BLSTMEncoder(num_layers=args.n_layers,
                                  hidden_size=args.hidden_size,
                                  embedding_depth=args.embedding_depth,
                                  bidirectional=args.bidirectional)
    timing_dic['Iitializing model'] = time.time() - before
    model = model.cuda()
    timing_dic['Transfering model to device'] = time.time() - before

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 betas=(0.9, 0.999))

    # just iterate over the data
    epochs = 10
    for epoch in np.arange(epochs):
        print("Training for epoch: {}...".format(epoch))
        for batch_data in training_generator:

            (abs_tfs, real_tfs, imag_tfs,
             duet_masks, ground_truth_masks,
             sources_raw, amplitudes, n_sources) = batch_data

            input_tfs, index_ys = abs_tfs.cuda(), duet_masks.cuda()
            # the input sequence is determined by time and not freqs
            # before: input_tfs = batch_size x (n_fft/2+1) x n_timesteps
            input_tfs = input_tfs.permute(0, 2, 1).contiguous()
            index_ys = index_ys.permute(0, 2, 1).contiguous()

            one_hot_ys = converters.one_hot_3Dmasks(index_ys, n_sources[0])

            timing_dic = {}

            optimizer.zero_grad()
            vs = model(input_tfs)

            before = time.time()
            flatened_ys = one_hot_ys.view(one_hot_ys.size(0),
                                          -1,
                                          one_hot_ys.size(-1)).cuda()
            naive_loss = affinity_losses.naive(vs, flatened_ys)
            naive_loss.backward()
            optimizer.step()
            now = time.time()
            print("Naive Loss: {}".format(naive_loss))
            timing_dic['Naive Loss Implementation Time'] = now - before

            optimizer.zero_grad()
            vs = model(input_tfs)

            before = time.time()
            expanded_vs = vs.view(vs.size(0), one_hot_ys.size(1),
                                  one_hot_ys.size(2), vs.size(-1)).cuda()
            diagonal_loss = affinity_losses.diagonal(expanded_vs,
                                                     one_hot_ys)
            diagonal_loss.backward()
            optimizer.step()
            now = time.time()
            print("Diagonal Loss: {}".format(diagonal_loss))
            timing_dic['Diagonal Loss Implementation Time'] = now - before

            pprint(timing_dic)



def get_args():
    """! Command line parser """
    parser = argparse.ArgumentParser(description='Deep Clustering for '
                                                 'Audio Source '
                                                 'Separation '
                                                 'Experiment')
    parser.add_argument("--dataset", type=str,
                        help="Dataset name",
                        default="timit")
    parser.add_argument("--n_sources", type=int,
                        help="How many sources in each mix",
                        default=2)
    parser.add_argument("--n_samples", type=int, nargs='+',
                        help="How many samples do u want to be "
                             "created for train test val",
                        default=[256, 64, 128])
    parser.add_argument("--genders", type=str, nargs='+',
                        help="Genders that will correspond to the "
                             "genders in the mixtures",
                        default=['m'])
    parser.add_argument("-f", "--force_delays", nargs='+', type=int,
                        help="""Whether you want to force integer 
                        delays of +- 1 in the sources e.g.""",
                        default=[-1, 1])
    parser.add_argument("-nl", "--n_layers", type=int,
                        help="""The number of layers of the LSTM 
                        encoder""", default=2)
    parser.add_argument("-ed", "--embedding_depth", type=int,
                        help="""The depth of the embedding""",
                        default=10)
    parser.add_argument("-hs", "--hidden_size", type=int,
                        help="""The size of the LSTM cells """,
                        default=10)
    parser.add_argument("-bs", "--batch_size", type=int,
                        help="""The number of samples in each batch""",
                        default=64)
    parser.add_argument("-name", "--experiment_name", type=str,
                        help="""The name or identifier of this 
                        experiment""",
                        default='A sample experiment')
    parser.add_argument("-cad", "--cuda_available_devices", type=int,
                        nargs="+",
                        help="""A list of Cuda IDs that would be 
                        available for runnign this experiment""",
                        default=[0])
    parser.add_argument("--num_workers", type=int,
                        help="""The number of cpu workers for 
                        loading the data, etc.""", default=3)
    parser.add_argument("-lr", "--learning_rate", type=float,
                        help="""Initial Learning rate""", default=1e-1)
    parser.add_argument("--bidirectional", action='store_true',
                        help="""Bidirectional or not""")

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    example_of_usage(args)