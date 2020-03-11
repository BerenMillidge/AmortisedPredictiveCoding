import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from .functions import *
from.models import *


#download and setup mnist dataset
trainset = torchvision.datasets.MNIST('MNIST_train', download=True, train=True)
valset = torchvision.datasets.MNIST('MNIST_test', download=True, train=False)
print("done")
print(trainset)
img = np.array(trainset[500][0])
print(trainset[500][1])
plt.imshow(img)
plt.show()

#Run initial testing experiments


"""fn = tanh
fn_deriv = tanhderiv
batch_size=10
num_batches = 10
n_inference_steps = 100
learning_rate= 0.01
layer_sizes = [784,300,100,10]
L = len(layer_sizes)
n_epochs = 20
imglist = [np.array([np.array(trainset[(n * batch_size) + i][0]).reshape([784,1]) / 255. for i in range(batch_size)]).T.reshape([784,batch_size]) for n in range(num_batches)]
labellist = [np.array([onehot(trainset[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_batches)]
prednet = PredictiveCodingNetwork(layer_sizes, learning_rate, batch_size,n_inference_steps, fn, fn_deriv)
prediction_errors = prednet.train(imglist, labellist, n_epochs)
prediction_errors = np.array(prediction_errors)
for i in range(1,L):
  print("Layer ", L - i, " prediction errors")
  plt.plot(prediction_errors[:,i])
  plt.show()"""

fn = tanh
fn_deriv = tanhderiv
q_fn = tanh
q_fn_deriv = tanhderiv
batch_size=10
num_batches = 10
n_inference_steps = 100
learning_rate= 0.01
amortised_learning_rate = 0.001
layer_sizes = [784,300,100,10]
L = len(layer_sizes)
n_epochs = 100
imglist = [np.array([np.array(trainset[(n * batch_size) + i][0]).reshape([784,1]) / 255. for i in range(batch_size)]).T.reshape([784,batch_size]) for n in range(num_batches)]
labellist = [np.array([onehot(trainset[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_batches)]
prednet = AmortisedPredictiveCodingNetwork(layer_sizes, learning_rate,amortised_learning_rate,batch_size, n_inference_steps, fn,fn_deriv,q_fn,q_fn_deriv)
prediction_errors,amortised_prediction_errors = prednet.train(imglist, labellist, n_epochs)
prediction_errors = np.array(prediction_errors)
amortised_prediction_errors = np.array(amortised_prediction_errors)
for i in range(1,L):
  print("Layer ", L - i, " prediction errors")
  plt.plot(prediction_errors[:,i])
  plt.show()
  print("Amortised prediction errors")
  plt.plot(amortised_prediction_errors[:,i-1])
  plt.show()
