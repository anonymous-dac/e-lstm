import torch
import math 
import numpy as np 


def set_to_zero_sparsity(inputs, sparsity=0.9):
    """
    args: 
    - inputs: the Tensor object of model weight 
    - sparsity: The sparsity we need 

    return:
    - the tensor with given sparsity
    """
    flatten = inputs.view(-1)
    absolute = torch.abs(flatten)
    sort = torch.sort(absolute)
    size = sort[0].size(0)
    threshold_index = int(math.floor(size * sparsity))
    threshold_value = sort[0][threshold_index]

    mask = torch.abs(inputs) > threshold_value
    mask = mask.float()
    return inputs * mask


def set_to_zero_threshold(inputs, threshold):
    """
    args: 
    - inputs: the Tensor object of model weight 
    - threshold: The threshold for zero

    return:
    - the tensor with given sparsity
    """
    absolute = torch.abs(inputs)
    mask = absolute > threshold
    mask = mask.float()
    return inputs * mask


def get_sparsity(inputs):
    """
    args: 
    - inputs: inputs tensor object 

    return:
        the sparsity of inputs
    """
    flatten = inputs.view(-1)
    absolute = torch.abs(flatten)
    total = absolute.size()[0]
    mask = absolute > 0
    non_zero = torch.sum(mask).numpy()
    return 1. - non_zero/total 

