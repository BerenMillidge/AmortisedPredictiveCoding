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

def accuracy(pred_labels, true_labels,print_error= False):
    #print("IN accuracy", pred_labels, true_labels)
    correct = 0
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
        correct +=1
      else:
        if print_error:
            print("PREDICTION: ",pred_labels[:,b] )
            print("LABEL: ",true_labels[:,b])
        else:
            pass

    return correct / batch_size

def same_sign(base, adjust):
  w,h = base.shape #currently only 2d arrays it's just a hacky prototype these things don't matter
  for i in range(w):
    for j in range(h):
      el = base[i,j]
      if el >0:
        adjust[i,j] = abs(adjust[i,j])
      else:
        adjust[i,j] = -abs(adjust[i,j])
  return adjust

def same_sign_binary(base, adjust):
  w,h = base.shape #currently only 2d arrays it's just a hacky prototype these things don't matter
  for i in range(w):
    for j in range(h):
      el = base[i,j]
      if el >0:
        adjust[i,j] = 1
      else:
        adjust[i,j] = -1
  return adjust

def dropout_mask(base_arr, dropout_prob):
  w,h = base_arr.shape
  drop = np.ones_like(base_arr)
  for i in range(w):
    for j in range(h):
      rand = np.random.uniform()
      if rand <= dropout_prob:
        drop[i,j] = 0
  return drop

#todo do a graph of how important sign concordance is here - for a simple weight transport tiny paper
# also similarly with and without synaptic weight adjustment too here. To replicate this paper effectively https://arxiv.org/pdf/1510.05067.pdf
def sign_concordance(base, adjust, concordance_prob):
    adjust = same_sign(base,adjust)
    w,h = adjust.shape
    for i in range(w):
        for j in range(h):
            r = np.random.uniform(low=0,high=1)
            if r >= concordance_prob:
                adjust[i,j] = -adjust[i,j]
    return adjust
