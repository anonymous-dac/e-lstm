from __future__ import print_function, absolute_import
import os,sys 
sys.path.append(os.getcwd())
import argparse
import torch 
import torch.nn as nn 
import torchvision 
import torchvision.transforms as transforms
from model import RNN
from utils.Param_transfer import set_to_zero_sparsity, get_sparsity, set_to_zero_threshold

sequence_length = 28 
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(1111)

def main(arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog="Mnist_eval", description="Evaluate network with different sparsity")
    parser.add_argument('-m', '--mode', required=True,
                        choices=['sparsity', 'threshold'], 
                        help="Whether to set sparsity or threshold")
    parser.add_argument('-ws', '--w_sparsity', type=float, 
                        default=0.5, help="Sparsity of weight")
    parser.add_argument('-hs', '--h_sparsity', type=float, 
                        default=0.6, help="The threshold for sparsity")
    parser.add_argument('-wt', '--threshold', type=float, 
                        default=0.1, help="The threshold for sparsity")
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help="Whether to print more information")
    parser.add_argument('--model_path', default='./MNIST/models/nhid:128-nlayer:2-epoch:20.ckpt', help="The path to saved model")

    args = parser.parse_args(arguments)
    #print (args.model_path)
    state_dict = torch.load(args.model_path)

    #sparsity_dict = {}

    if args.mode == 'sparsity':
        for k, v in state_dict.items():
            state_dict[k] = set_to_zero_sparsity(v, sparsity=args.w_sparsity)
    else:
        for k, v in state_dict.items():
            state_dict[k] = set_to_zero_threshold(v, threshold=args.threshold)

    
    test_dataset = torchvision.datasets.MNIST(root='./MNIST/data/', train=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    model = RNN(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(state_dict)
    model.half()
    # states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
    #           torch.zeros(num_layers, batch_size, hidden_size).to(device))

    with torch.no_grad():
        correct = 0
        total = 0
        hidden = model.init_hidden(batch_size)
        for images, labels in test_loader:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            outputs, hidden = model(images, hidden, args.h_sparsity)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        result = (args.w_sparsity, args.h_sparsity, accuracy)
        if args.verbose:
            print('|Weight sparsity: {:.3f}| hidden state sparsity: {:.3f} | Test Accuracy : {:.5f} |'.format(args.w_sparsity, args.h_sparsity, accuracy)) 
        else:
            print (result)

if __name__ == '__main__':
    main()