import os, sys
import torch
import math 
import numpy as np 
import argparse
from utils.Param_transfer import set_to_zero_threshold, set_to_zero_sparsity, get_sparsity 

"""
Get the distibution of a trained model 
"""

def get_statics(data):
    """
    arguments: 
        - data: A numpy array
    
    return:
        - a tuple of statics
    """
    data = np.absolute(data.flatten())

    d_min = np.amin(data)
    d_max = np.amax(data)
    mean = np.mean(data)
    std = np.std(data)

    percent = [25, 50, 75, 90]
    percent_data = np.percentile(data, percent)

    return (mean, std, d_min, percent_data, d_max)
    

def main(arguments=sys.argv[1:]):
    parser = argparse.ArgumentParser(prog="Model data", description="Evaluate the distribution of trained model")
    parser.add_argument('-t', '--threshold', type=float, default=0.01,
                        help='Threshold for sparsity calculation')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                        help='The model path')
    args = parser.parse_args(arguments)

    state_dict = torch.load(args.model_path, map_location = lambda storage, loc:storage)

    #statics = {}
    print("|{:<30}|{:<9}|{:<9}|{:<9}|{:<9}|{:<9}|{:<9}|{:<9}|{:<9}|".format(
        'name', "mean", "std", "min", "25%", "50%", "75%", "90%", "max"
    ))
    for k, v in state_dict.items():
        data = v.cpu().numpy()
        statics = get_statics(data)
        print("|{:<30}|{:<9.3e}|{:<9.3e}|{:<9.3e}|{:<9.3e}|{:<9.3e}|{:<9.3e}|{:<9.3e}|{:<9.3e}|".format(
            k, statics[0], statics[1], statics[2], statics[3][0], statics[3][1], statics[3][2], statics[3][3], statics[4]
        ))



if __name__ == "__main__":
    main()