# Word-level language modeling RNN

## General description and usage

This example trains a multi-layer RNN(LSTM) on a language modeling task.
By default, the training script uses the PTB dataset, provided.
The trained model can then be used by the generate script to generate new text.

```bash
python main.py --cuda --epochs 6 --tied  # 6 epoch 
python main.py --cuda --tied             # 40 epoch                   
```

The model used custimized LSTM implementation.
During training, if a keyboard interrupt (Ctrl-C) is received,
training is stopped and the current model is evaluated against the test dataset.

The `main.py` script accepts the following arguments:

```bash
optional arguments:
  -h, --help         show this help message and exit
  --data DATA        location of the data corpus
  --model MODEL      type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --emsize EMSIZE    size of word embeddings
  --nhid NHID        number of hidden units per layer
  --nlayers NLAYERS  number of layers
  --lr LR            initial learning rate
  --clip CLIP        gradient clipping
  --epochs EPOCHS    upper epoch limit
  --batch-size N     batch size
  --bptt BPTT        sequence length
  --dropout DROPOUT  dropout applied to layers (0 = no dropout)
  --decay DECAY      learning rate decay per epoch
  --tied             tie the word embedding and softmax weights
  --seed SEED        random seed
  --cuda             use CUDA
  --l1               Whether to add l1 regularization 
  --l1_lambda        The lambda value of l1 regularization
  --log-interval N   report interval
  --save_dir         path to save the final model
  --log_dir          path to store the log files
```

With these arguments, a variety of models can be tested.
As an example, the following arguments produce slower but better models:

```bash       
python main.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40 --tied     # Medium size       
python main.py --cuda --emsize 1500 --nhid 1500 --dropout 0.65 --epochs 40 --tied  # Large size 
```


## Evaluation 
```
python LM/eval.py -m sparsity -v --h_threshold threshold
```