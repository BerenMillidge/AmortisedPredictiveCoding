import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from functions import *
from models import *

def run_amortised(trainset, valset):
    fn = tanh
    fn_deriv = tanhderiv
    qf = tanh
    qf_deriv = tanhderiv
    batch_size=10
    num_batches = 10
    n_inference_steps = 100
    learning_rate= 0.01
    amortised_learning_rate = 0.001
    layer_sizes = [784,300,100,10]
    L = len(layer_sizes)
    n_epochs = 101
    imglist = [np.array([np.array(trainset[(n * batch_size) + i][0]).reshape([784,1]) / 255. for i in range(batch_size)]).T.reshape([784,batch_size]) for n in range(num_batches)]
    labellist = [np.array([onehot(trainset[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_batches)]
    prednet = AmortisedPredictiveCodingNetwork(layer_sizes, batch_size,learning_rate,amortised_learning_rate, fn, fn_deriv,qf, qf_deriv,n_inference_steps)
    prediction_errors = prednet.train(imglist, labellist, n_epochs)

if __name__ == "__main__":
    trainset = torchvision.datasets.MNIST('MNIST_train', download=True, train=True)
    valset = torchvision.datasets.MNIST('MNIST_test', download=True, train=False)
    run_amortised(trainset, valset)
