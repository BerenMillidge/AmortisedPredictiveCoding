import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from functions import *
from models import *
import sys

def run_amortised(save_name):
    batch_size = 50
    num_batches = 10
    num_test_batches = 20
    n_inference_steps_train = 100
    n_inference_steps_test = 1000
    learning_rate = 0.01
    amortised_learning_rate = 0.001
    layer_sizes = [784, 300, 100, 10]
    n_layers = len(layer_sizes)
    n_epochs = 101
    inference_thresh = 0.5

    train_set = torchvision.datasets.MNIST("MNIST_train", download=True, train=True)
    test_set = torchvision.datasets.MNIST("MNIST_test", download=True, train=False)
    num_batches = len(train_set)// batch_size
    print("Num Batches",num_batches)
    img_list = [
        np.array(
            [
                np.array(train_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0
                for i in range(batch_size)
            ]
        ).T.reshape([784, batch_size])
        for n in range(num_batches)
    ]
    label_list = [
        np.array([onehot(train_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T
        for n in range(num_batches)
    ]
    test_img_list = [
        np.array(
            [
                np.array(test_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0
                for i in range(batch_size)
            ]
        ).T.reshape([784, batch_size])
        for n in range(num_test_batches)
    ]
    test_label_list = [
        np.array([onehot(test_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T
        for n in range(num_test_batches)
    ]
    pred_net = AmortisedPredictiveCodingNetwork(
        layer_sizes,
        batch_size,
        learning_rate,
        amortised_learning_rate,
        n_inference_steps_train=n_inference_steps_train,
        n_inference_steps_test=n_inference_steps_test,
        f=tanh,
        df=tanhderiv,
        qf=tanh,
        dqf=tanhderiv,
        inference_threshold=inference_thresh
    )
    pred_net.train(img_list, label_list,test_img_list, test_label_list, n_epochs,save_name)

if __name__ == "__main__":
    sname = str(sys.argv[1])
    run_amortised(sname)
