#functions
from scipy.special import expit
from copy import deepcopy
import numpy as np


def linear(x):
  return x

def linearderiv(x):
  return 1

def sigmoid(xs):
  return expit(xs)

def sigmoid2(xs):
  return 1 / (1 +  np.exp(-xs))

def sigmoidderiv(xs):
  return sigmoid(xs) * (1 - sigmoid(xs))

def relu(xs):
    return np.maximum(xs,0,xs)

def reluderiv(xs):
    rel = relu(xs)
    rel[rel>0] = 1
    return rel

def tanh(xs):
  return np.tanh(xs)

def tanhderiv(xs):
  return 1 - np.tanh(xs)**2

def onehot(x):
    z = np.zeros([10,])
    z[x] = 1
    return z

def accuracy(pred_labels, true_labels):
    print("IN accuracy", pred_labels, true_labels)
    correct = 0
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
        correct +=1
    return correct / batch_size
