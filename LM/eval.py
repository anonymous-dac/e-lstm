from __future__ import print_function, absolute_import
import os,sys 
sys.path.append(os.getcwd())
import argparse
import math

import torch
import torch.nn as nn
import torch.onnx
import data
import model
#from model import RNNModel
from utils.Param_transfer import set_to_zero_sparsity, set_to_zero_threshold
#from config import SmallConfig, MediumConfig, LargeConfig


RNN_type = "LSTM"
emsize = 1500
nhid = 1500
nlayers = 2 
dropout = 0.2 
bptt = 35
tied = False
eval_batch_size = 10
criterion = nn.CrossEntropyLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(model, data_source, ntokens, h_sparsity=0., h_threshold=0.):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    #ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden, h_sparsity, h_threshold)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / (len(data_source) - 1)
 

def main(arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog="LM_eval", description="Evaluate network with different sparsity")
    parser.add_argument('-m', '--mode', required=True,
                        choices=['sparsity', 'threshold'], 
                        help="Whether to set sparsity or threshold")
    parser.add_argument('--data', type=str, default='./LM/data/PTB/',
                        help='location of the data corpus')
    parser.add_argument('-ws', '--w_sparsity', type=float, 
                        default=0., help="Sparsity of weight")
    parser.add_argument('-hs', '--h_sparsity', type=float, 
                        default=0., help="The threshold for sparsity")
    parser.add_argument('-wt', '--w_threshold', type=float, 
                        default=0.1, help="The threshold for sparsity")
    parser.add_argument('-ht', '--h_threshold', type=float, 
                        default=0.1, help="The threshold for sparsity")
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help="Whether to print more information")
    parser.add_argument('--model_path', default='./LM/models/PTB/model:LSTM-em:1500-nhid:1500-nlayers:2-bptt:35.ckpt', help="The path to saved model")
    parser.add_argument('--seed', type=int, default=1111, help='random seed')
    parser.add_argument('--cuda', action='store_true', help="Whether to use cuda")
    args = parser.parse_args(arguments)

    
    state_dict = torch.load(args.model_path, map_location = lambda storage, loc:storage)

    torch.manual_seed(args.seed)
    # if torch.cuda.is_available():
    #     if not args.cuda:
    #         print("WARNING: You have a CUDA device, so you should probably run with --cuda")


    corpus = data.Corpus(args.data)
    test_data = batchify(corpus.test, eval_batch_size)

    # Add weight sparsity
    if args.mode == 'sparsity':
        for k, v in state_dict.items():
            if 'lstm' in k:
                state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sparsity)
    else:
        for k, v in state_dict.items():
            if 'lstm' in k:
                state_dict[k] = set_to_zero_threshold(v, threshold=args.w_threshold)

    
    ntokens = len(corpus.dictionary)
    test_model = model.RNNModel(RNN_type, ntokens, emsize, nhid, nlayers, dropout, tied).to(device)
    test_model.load_state_dict(state_dict)
    test_model.half()
    test_loss = evaluate(test_model, test_data, ntokens, h_threshold=args.h_threshold)
    print("|Evaluate on test data| loss: {:5.2f}| PPL {:8.2f}|".format(
        test_loss, math.exp(test_loss)
    ))
    for k, v in test_model.hidden_sparsity.items():
        print("|{} | num_batch: {}| sparsity {:5.4f}|".format(k, len(v), sum(v)/float(len(v))))
   


if __name__ == '__main__':
    main()