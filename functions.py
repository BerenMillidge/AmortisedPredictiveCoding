#functions
from scipy.special import expit
from copy import deepcopy

def relu(x):
  if x>0:
    return x
  return 0

def linear(x):
  return x

def linearderiv(x):
  return 1

def reluderiv(x):
  if x >0:
    return 1
  return 0

def sigmoid(xs):
  return expit(xs)

def sigmoid2(xs):
  return 1 / (1 +  np.exp(-xs))

def sigmoidderiv(xs):
  return sigmoid(xs) * (1 - sigmoid(xs))

def tanh(xs):
  return np.tanh(xs)

def tanhderiv(xs):
  return 1 - np.tanh(xs)**2

def onehot(x):
    z = np.zeros([10,])
    z[x] = 1
    return z
