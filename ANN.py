# standard backprop ANN in pytorch to compare to

import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from functions import *
from transformer import *
import sys
import torchvision
import torch.nn.functional as F
import torch.nn as nn
# define ANN model
class ANN(nn.Module):

    def __init__(self, layer_sizes, act_fn=F.tanh,learning_rate=1e-3,epsilon=1e-4):
        self.layer_sizes= layer_sizes
        self.act_fn = act_fn
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.layers = [nn.Linear(self.layer_sizes[i],self.layer_sizes[i+1]) for i in range(len(self.layer_sizes)-1)]
        self.params = []
        for l in self.layers:
            self.params += list(l.parameters())
        self.optim = torch.optim.Adam(self.params, lr=self.learning_rate, eps = self.epsilon)

    def forward(self, x):
        #print(x.shape)
        for l in self.layers:
            x = self.act_fn(l(x))
        return x

    def test(self, x):
        return self.forward(x)

    # so this used to work, and I have NO idea what I did that stopped it working...
    #it's really odd actually. I mean I have PROOF that this used to get to 1
    #and I have no idea why it stopped

    def train(self, imglist, labellist,test_img_list, test_label_list, n_epochs):
        for n in range(n_epochs):
          print("Epoch: ", n)
          for (img_batch,label_batch) in zip(imglist, labellist):
            img_batch = torch.tensor(img_batch, dtype=torch.float32).permute(1,0)
            label_batch = torch.tensor(label_batch,dtype=torch.float32).permute(1,0)
            #print(img_batch.size())
            self.optim.zero_grad()
            #print("IN OPTIM!")
            preds = self.forward(img_batch)
            #print("AFTER FORWARD?")
            #loss = F.mse_loss(preds, label_batch)
            loss = torch.sum((preds - label_batch)**2)
            #print("PREDICTION: ", preds.detach().numpy()[0,:])
            #print("LABEL: ", label_batch[0,:].numpy())
            #print("LOSS: ", loss.item())
            loss.backward()
            #for p in self.params:
            #    print(p.shape)
            #    print(p.mean().item())
            torch.nn.utils.clip_grad_norm_(self.params, 1000,norm_type=2)
            self.optim.step()

          if n % 10 == 0:
            tot_acc = 0
            for (img_batch, label_batch) in zip(imglist, labellist):
              img_batch = torch.tensor(img_batch, dtype=torch.float32).permute(1,0)
              #label_batch = torch.tensor(label_batch,dtype=torch.float32)
              pred_labels = self.test(img_batch).permute(1,0).detach().numpy()
              #print("pred :", pred_labels.shape)
              #print("label: ",label_batch.shape)

              #print("PREDICTIONS: ", pred_labels[0,:])
              #print("LABELS: ", label_batch[0,:])
              tot_acc += accuracy(pred_labels, label_batch)
              #print("ACCURACY: ", accuracy(pred_labels, label_batch))
            print("Accuracy: ", tot_acc/len(imglist))
            tot_acc = 0
            for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
                test_img_batch = torch.tensor(test_img_batch, dtype=torch.float32).permute(1,0)
                pred_labels = self.test(test_img_batch).permute(1,0).detach().numpy()
                tot_acc += accuracy(pred_labels, test_label_batch)
            print(f"Test Accuracy: {tot_acc/len(test_img_list)}")

    def test_occluded(self,test_img_list, test_label_list,save_name):
        num_occluders = [1,3,5,7,9]
        n_occluder_accuracy_list = []
        for (i,n_occ) in enumerate(num_occluders):
            tot_acc = 0
            for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
                occluded_batch = apply_batch(test_img_batch,occlude,(5,5),n_occ,0)
                occluded_batch = torch.tensor(occluded_batch, dtype=torch.float32).permute(1,0)
                pred_labels = self.test(occluded_batch).permute(1,0).detach().numpy()
                tot_acc += accuracy(pred_labels, test_label_batch,print_error=True)

            n_occluder_accuracy_list.append(tot_acc / len(test_img_list))
        n_occluder_accuracy_list = np.array(n_occluder_accuracy_list)
        print("OCCLUDER ACCURACY LIST SIZE: ", n_occluder_accuracy_list.shape)
        np.save(save_name + "_n_occluder_acc_list.npy",np.array(n_occluder_accuracy_list))
        fig = plt.figure()
        plt.title("Accuracy per number of occluders")
        plt.bar(num_occluders, n_occluder_accuracy_list)
        plt.ylabel("Accuracy")
        plt.xlabel("num occluders")
        plt.show()

        occluder_sizes = [1,2,3,4,5,6,7]
        occluder_size_accuracy_list = []
        for (i,occ_size) in enumerate(occluder_sizes):
            tot_acc = 0
            for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
                occluded_batch = apply_batch(test_img_batch,occlude,(occ_size,occ_size),5,0)
                occluded_batch = torch.tensor(occluded_batch, dtype=torch.float32).permute(1,0)
                pred_labels = self.test(occluded_batch).permute(1,0).detach().numpy()
                print("PRED LABELS: ", pred_labels.shape, test_label_batch.shape)
                tot_acc += accuracy(pred_labels, test_label_batch,print_error=True)
            occluder_size_accuracy_list.append(tot_acc / len(test_img_list))
        occluder_size_accuracy_list = np.array(occluder_size_accuracy_list)
        np.save(save_name + "_occluder_size_acc_list.npy",np.array(occluder_size_accuracy_list))
        # plot
        fig = plt.figure()
        plt.title("Accuracy against occluder size")
        plt.bar(num_occluders, n_occluder_accuracy_list)
        plt.ylabel("Accuracy")
        plt.xlabel("num occluders")
        plt.show()






def run_ANN(save_name):
    batch_size = 20
    num_batches = 100
    num_test_batches = 10
    learning_rate = 0.0001
    layer_sizes = [784, 300, 100, 10]
    n_layers = len(layer_sizes)
    n_epochs = 101
    train_set = torchvision.datasets.MNIST("MNIST_train", download=True, train=True)
    test_set = torchvision.datasets.MNIST("MNIST_test", download=True, train=False)
    #num_batches = len(train_set)// batch_size
    print("Num Batches",num_batches)
    img_list = [np.array([np.array(train_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0 for i in range(batch_size)]).T.reshape([784, batch_size]) for n in range(num_batches)]
    label_list = [np.array([onehot(train_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_batches)]
    test_img_list = [np.array([np.array(test_set[(n * batch_size) + i][0]).reshape([784, 1]) / 255.0 for i in range(batch_size)]).T.reshape([784, batch_size]) for n in range(num_test_batches)]
    test_label_list = [np.array([onehot(test_set[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_test_batches)]

    net = ANN(
        layer_sizes,
        act_fn = F.tanh,
        learning_rate=learning_rate,
    )
    net.train(img_list, label_list,test_img_list, test_label_list, n_epochs)
    net.test_occluded(test_img_list, test_label_list,save_name)

if __name__ == "__main__":
    sname = str(sys.argv[1])
    run_ANN(sname)
