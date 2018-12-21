# OCR 
LSTM is widely used together with CNN in OCR. In this project, we use MNIST and purely LSTM for simplisity cause we aim to accelerating the LSTM.


## Train

```
python MNIST/train.py 
```
If you want to change the hyperparameters, just change those values directly in MNIST/train.py

## Evaluate

To evaluate the trained model  
```
python MNIST/eval.py -m sparsity -v --model_path /path/to/trained_model
```

To get the help information about the argument 
```
Python MNIST/eval.py -h
```
