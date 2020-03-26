import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
import sys
import torchvision
from scipy import ndimage

def translate(img,px,mode='constant'):
    return ndimage.interpolation.shift(img, px, mode=mode)

def rotate(img, angle, reshape=False):
    return ndimage.rotate(img, angle, reshape=reshape)

def occlude(img, occluder_size,n_occluders, occluder_pxval=1):
    h,w = occluder_size
    ih,iw = img.shape
    n = 0
    while n < n_occluders:
        occ_h = int(np.random.uniform(low=0,high=ih))
        occ_w = int(np.random.uniform(low=0,high=iw))
        if occ_h < ih -h and occ_w < iw -w:
            img[occ_h:occ_h+h,occ_w:occ_w+w] = np.ones([h,w]) * occluder_pxval
            n+=1
    return img

def onehot(x):
    out = np.zeros(10)
    out[x] = 1
    return out


def apply_batch(batch, f, *args):
    out = np.zeros_like(batch)
    for b in range(batch.shape[1]):
        transformed = f(batch[:,b].reshape(28,28),*args)
        out[:,b] = transformed.reshape(784)
    return out


if __name__ == '__main__':
    num_batches = 10
    batch_size = 10
    train_set = torchvision.datasets.MNIST("MNIST_train", download=True, train=True)
    test_set = torchvision.datasets.MNIST("MNIST_test", download=True, train=False)
    #num_batches = len(train_set)// batch_size
    print("Num Batches",num_batches)
    img_list = [np.array([np.array(train_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0 for i in range(batch_size)]).T.reshape([784, batch_size]) for n in range(num_batches)]
    label_list = [np.array([onehot(train_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_batches)]
    #test_img_list = [np.array(np.array(test_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0 for i in range(batch_size)]).T.reshape([784, batch_size]) for n in range(num_test_batches)]
    #test_label_list = [np.array([onehot(test_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_test_batches)]
    t = apply_batch(img_list[0],rotate, 45)
    print(t.shape)
    for i in range(10):
        plt.imshow(t[:,i].reshape(28,28))
        plt.show()
