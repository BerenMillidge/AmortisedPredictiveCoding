# generic place to put together ALL biological plausibility experiments which I have operational so far.
#condense all options down into a single file to make running unified experiment suites easier.

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from functions import *
import sys
import torchvision
import os
import time
import subprocess
from datetime import datetime
import argparse

class PredictiveCodingLayer(object):
  def __init__(self,input_size, output_size, fn,fn_deriv,args):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = args.batch_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.learning_rate = args.learning_rate
    self.use_backward_weights = args.weight_symmetry
    self.dropout_prob = args.dropout_prob
    self.sign_concordance_prob = args.sign_concordance_prob
    self.use_backward_nonlinearity = args.use_backward_nonlinearity
    self.weights = np.random.normal(0,0.1,[input_size, output_size])
    if not self.use_backward_weights:
        self.backward_weights = deepcopy(self.weights.T)
    else:
        if self.sign_concordance_prob is not None:
            self.backward_weights = sign_concordance(self.weights.T, np.random.normal(0,0.1,[input_size,output_size]).T,self.sign_concordance_prob)
        else:
            self.backward_weights = np.random.normal(0,0.1,[input_size, output_size]).T
        # do dropout if required
        if self.dropout_prob is not None:
            self.weight_mask = dropout_mask(self.weights, self.dropout_prob)
            self.backward_weight_mask = dropout_mask(self.backward_weights,self.dropout_prob)
            self.weights *= self.weight_mask
            self.backward_weights *= self.backward_weight_mask
    print(input_size, self.batch_size)
    self.mu = np.random.normal(0,1,[output_size,self.batch_size])

  def update_mu(self, pe, pe_below):
    if self.use_backward_nonlinearity:
        self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below )))
    else:
        self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below)))
  def step(self,pe_below,pred,use_top_down_pe=True):
    if use_top_down_pe:
      pe = self.mu - pred
    else:
      pe = np.zeros_like(self.mu)
    if self.use_backward_nonlinearity:
        self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below )))
    else:
        self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below)))
    return pe, self.predict()

  def predict(self):
    return self.fn(np.dot(self.weights, self.mu))

  def update_weights(self,pe_below):
    if self.use_backward_nonlinearity:
        self.weights += self.learning_rate * (np.dot(pe_below * self.fn_deriv(np.dot(self.weights, self.mu)), self.mu.T))
    else:
        self.weights += self.learning_rate * (np.dot(pe_below, self.mu.T))
    #reapply dropout mask
    if self.dropout_prob is not None:
        self.weights *= self.weight_mask


  def update_backward_weights(self, pe_below):
    if self.use_backward_nonlinearity:
        self.backward_weights += self.learning_rate * np.dot(self.mu,(pe_below * self.fn_deriv(np.dot(self.weights, self.mu))).T)
    else:
        self.backward_weights += self.learning_rate * np.dot(self.mu,pe_below.T)
    #reapply dropout mask
    if self.dropout_prob is not None:
        self.backward_weights *= self.backward_weight_mask

class PredictiveCodingNetwork(object):

  def __init__(self, layer_sizes, f,df,args):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.n_inference_steps = n_inference_steps
    self.layers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.f,self.df,args))
    self.predictions = [[]for i in range(len(self.layer_sizes))]
    self.prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.L = len(self.layers)

  def reset_mus(self):
    # to prevent horrible and weird state corruption/dependencies between trials which give all manner of lovely bugs.
    for layer in self.layers:
        layer.mu = np.random.normal(0,1,[layer.output_size,layer.batch_size])

  def infer(self, img_batch,label_batch):
    self._reset_mus()
    prediction_errors = [[] for i in range(self.L)]
    self.layers[-1].mu = deepcopy(label_batch)
    for i in reversed(range(self.L)):
      self.predictions[i] = self.layers[i].predict()
    for n in range(self.n_inference_steps):
      self.prediction_errors[0] = img_batch - self.predictions[0]
      self.predictions[-1] = np.zeros_like(self.layers[-1].mu)
      for l in range(self.L):
        self.prediction_errors[l+1] = self.layers[l].mu - self.predictions[l+1]
        if l !=self.L-1:
          self.layers[l]._update_mu(self.prediction_errors[l+1],self.prediction_errors[l])
        else:
          self.layers[l]._update_mu(np.zeros_like(self.layers[l].mu), self.prediction_errors[l])
        self.predictions[l] = self.layers[l].predict()
      self.layers[-1].mu = deepcopy(label_batch)
      self.predictions[-1] = self.layers[-1].mu

    for l in range(self.L):
      self.layers[l].update_weights(self.prediction_errors[l])
    return self.prediction_errors, self.predictions



  def test(self, img_batch, label_batch):
    self._reset_mus()
    prediction_errors = [[] for i in range(self.L)]
    for i in reversed(range(self.L)):
      self.predictions[i] = self.layers[i].predict()
    for n in range(self.n_inference_steps * 10):
      self.prediction_errors[0] = img_batch - self.predictions[0]
      self.predictions[-1] = np.zeros_like(self.layers[-1].mu)
      for l in range(self.L):
        self.prediction_errors[l+1] = self.layers[l].mu - self.predictions[l+1]
        if l !=self.L-1:
          self.layers[l]._update_mu(self.prediction_errors[l+1],self.prediction_errors[l])
        else:
          self.layers[l]._update_mu(np.zeros_like(self.layers[l].mu), self.prediction_errors[l])
        self.predictions[l] = self.layers[l].predict()


    pred_labels = self.layers[-1].mu
    pred_imgs = self.predictions[0]
    return pred_imgs, pred_labels

  def plot_batch_results(self,pred_labels,true_labels):
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      plt.plot(pred_labels[:,b])
      plt.show()
      print(true_labels[:,b])


  def train(self, imglist, labellist, n_epochs):
    prediction_errors = []
    for n in range(n_epochs):
      print("Epoch ", n)
      batch_pes = []
      for (img_batch,label_batch) in zip(imglist, labellist):
        pes, preds = self.infer(img_batch,label_batch)

      if n % 10 == 0:
        tot_acc = 0
        for (img_batch, label_batch) in zip(imglist, labellist):
          pred_imgs, pred_labels = self.test(img_batch,label_batch)
          tot_acc += self.accuracy(pred_labels, label_batch)
        print("Accuracy: ", tot_acc/num_batches)

class AmortisationLayer(object):
  def __init__(self, forward_size, backward_size, learning_rate, batch_size, qf, dqf,use_backward_nonlinearity):
    self.forward_size = forward_size
    self.backward_size = backward_size
    self.learning_rate = learning_rate
    self.batch_szie = batch_size
    self.use_backward_nonlinearity = use_backward_nonlinearity
    self.qf = qf
    self.dqf = dqf
    self.weights = np.random.normal(0,0.1,[self.forward_size, self.backward_size])

  def predict(self, state):
    return self.qf(np.dot(self.weights, state))


  def update_weights(self, amortised_prediction_errors, state):
    if self.use_backward_nonlinearity:
        self.weights += self.learning_rate * (np.dot(amortised_prediction_errors * self.dqf(np.dot(self.weights, state)),state.T))
    else:
        self.weights += self.learning_rate * (np.dot(amortised_prediction_errors,state.T))

class AmortisedPredictiveCodingNetwork(object):
  def __init__(self, layer_sizes, f,df,qf,dqf,args):
    self.layer_sizes = layer_sizes
    self.batch_size = args.batch_size
    self.learning_rate = args.learning_rate
    self.amortised_learning_rate = args.amortised_learning_rate
    self.n_inference_steps_train = args.n_inference_steps_train
    self.n_inference_steps_test = args.n_inference_steps_test
    self.dropout_prob = args.dropout_prob
    self.sign_concordance_prob = args.sign_concordance_prob
    self.use_backward_nonlinearity = args.use_backward_nonlinearity
    self.weight_symmetry = args.weight_symmetry
    self.backward_weight_update = args.backward_weight_update
    self.save_path = args.save_path
    self.log_path = args.log_path
    self.save_every = args.save_every
    self.n_epochs = args.n_epochs
    self.f = f
    self.df = df
    self.qf = qf
    self.dqf = dqf
    self.inference_threshold = args.inference_threshold
    self.layers = []
    self.q_layers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.f,self.df,args))
    #initialize amortised networks
    for i in range(len(self.layer_sizes)-1):
      self.q_layers.append(AmortisationLayer(self.layer_sizes[i+1], self.layer_sizes[i], self.amortised_learning_rate,self.batch_size, self.qf,self.dqf,self.use_backward_nonlinearity))
    self.predictions = [[]for i in range(len(self.layer_sizes))]
    self.prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.amortised_predictions = [[] for i in range(len(self.layer_sizes))]
    self.amortised_prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.L = len(self.layers)
    self.n_layers = len(self.layer_sizes)

  def reset_mus(self):
    for layer in self.layers:
        layer.mu = np.random.normal(0,1,[layer.output_size,layer.batch_size])



  def amortisation_pass(self, img_batch,initialize=True):
      self.amortised_predictions[0] = deepcopy(img_batch)
      for i in range(self.L):
          self.amortised_predictions[i+1] = self.q_layers[i].predict(self.amortised_predictions[i])
          self.layers[i].mu = deepcopy(self.amortised_predictions[i+1])



  def forward_pass(self, img_batch, label_batch, test=False):
    # reset model
    n_inference_steps = self.n_inference_steps_test if test else self.n_inference_steps_train
    self.reset_mus()
    prediction_errors = [[] for i in range(self.n_layers)]

    #set the highest level mus to the label if training. Else let them vary freely (they will become the prediction).
    #run an amortisation pass to initialize all variational parameters.
    self.amortisation_pass(img_batch,initialize=True)
    if not test:
        self.layers[-1].mu = deepcopy(label_batch)

    # variational predictions (predictions go top down so have to go through layers in reverse order)
    for i in reversed(range(self.L)):
        self.predictions[i] = self.layers[i].predict()

    # perform variational updates
    for n in range(n_inference_steps):
        # set lowest prediction errors from the input stimulus
        self.prediction_errors[0] = img_batch - self.predictions[0]
        #set highest predictions to zero since there are no top-down predictions entering the highest layer
        self.predictions[-1] = np.zeros_like(self.layers[-1].mu)
        #perform variational updates for each layer
        for i in range(self.L):
            self.prediction_errors[i+1] = self.layers[i].mu - self.predictions[i+1]
            if i != self.L-1:
                self.layers[i].update_mu(self.prediction_errors[i+1], self.prediction_errors[i])
            else:
                self.layers[i].update_mu(np.zeros_like(self.layers[i].mu), self.prediction_errors[i])
            self.predictions[i] = self.layers[i].predict()
            #set reset the final layer mus to the batch label
        if not test:
            self.layers[-1].mu = deepcopy(label_batch)
            self.predictions[-1] = deepcopy(self.layers[-1].mu)


        F = np.sum(np.array([np.mean(np.square(pe)) for pe in self.prediction_errors]))
        if F <= self.inference_threshold:
            break

    pred_imgs = deepcopy(self.predictions[0])
    pred_labels = deepcopy(self.layers[-1].mu)
    return pred_imgs, pred_labels

  def train_batch(self, img_batch, label_batch):
    self.forward_pass(img_batch, label_batch, test=False)
    for l in range(self.L):
        self.layers[l].update_weights(self.prediction_errors[l])
        #only update backward weights if true. If false no updates - i.e. feedback alignment tests and variance
        if self.backward_weight_update:
            self.layers[l].update_backward_weights(self.prediction_errors[l])
        self.amortised_prediction_errors[l] = self.layers[l].mu - self.amortised_predictions[l+1]
        if l == 0:
            self.q_layers[l].update_weights(self.amortised_prediction_errors[l], self.amortised_predictions[0])
        else:
            self.q_layers[l].update_weights(self.amortised_prediction_errors[l], self.layers[l-1].mu)

    return self.prediction_errors, self.predictions,self.amortised_predictions, self.amortised_prediction_errors


  def test(self, img_batch, label_batch):
    pred_imgs, pred_labels  = self.forward_pass(img_batch, label_batch,test=True)
    return pred_imgs, pred_labels

  def amortised_test(self, img_batch):
    self.amortisation_pass(img_batch,initialize=True)
    return self.amortised_predictions[-1]

  def plot_batch_results(self,pred_labels,amortised_labels, true_labels):
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      print("True labels: ", true_labels[:,b])
      print("Variational predictions: ")
      plt.plot(pred_labels[:,b])
      plt.show()
      print("Amortised Predictions: ",)
      plt.plot(amortised_labels[:,b])
      plt.show()

  def accuracy(self,pred_labels, true_labels):
      #print("IN accuracy", pred_labels, true_labels)
      correct = 0
      batch_size = pred_labels.shape[1]
      for b in range(batch_size):
        if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
          correct +=1
      return correct / batch_size


  def train(self, imglist, labellist,test_img_list, test_label_list):
    prediction_errors = []
    amortised_prediction_errors = []
    variational_accs = []
    amortised_accs = []
    test_variational_accs = []
    test_amortised_accs = []
    for n in range(self.n_epochs):
        print("Epoch ", n)
        batch_pes = []
        batch_qpes = []
        for (img_batch,label_batch) in zip(imglist, labellist):
            pes, preds,qpreds, qpes = self.train_batch(img_batch,label_batch)
            batch_pes.append(np.array([np.sum(pe) for pe in pes]))
            batch_qpes.append(np.array([np.sum(qpe) for qpe in qpes]))

        prediction_errors.append(np.array(batch_pes))
        amortised_prediction_errors.append(np.array(batch_qpes))

        if n % self.save_every == 0:
            tot_acc = 0
            q_acc = 0
            for (img_batch, label_batch) in zip(imglist, labellist):
              pred_imgs, pred_labels = self.test(img_batch,label_batch)
              tot_acc += self.accuracy(pred_labels, label_batch)
              pred_qlabels = self.amortised_test(img_batch)
              q_acc += self.accuracy(pred_qlabels, label_batch)
            print("Accuracy: ", tot_acc/len(imglist))
            print("Amortised Accuracy: ", q_acc / len(imglist))
            variational_accs.append(tot_acc/len(imglist))
            amortised_accs.append(q_acc / len(imglist))
            print("TEST ACCURACIES")
            tot_acc = 0
            q_acc = 0
            for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
                pred_imgs, pred_labels = self.test(test_img_batch, test_label_batch)
                tot_acc += accuracy(pred_labels, test_label_batch)
                pred_qlabels = self.amortised_test(test_img_batch)
                q_acc += accuracy(pred_qlabels, test_label_batch)
            print(f"Test Variational Accuracy: {tot_acc/len(test_img_list)}")
            print(f"Test Amortised Accuracy: {q_acc / len(test_img_list)}")
            test_variational_accs.append(tot_acc/len(test_img_list))
            test_amortised_accs.append(q_acc / len(test_img_list))
            np.save(self.log_path + "_variational_acc.npy", np.array(deepcopy(variational_accs)))
            np.save(self.log_path + "_amortised_acc.npy", np.array(deepcopy(amortised_accs)))
            np.save(self.log_path + "_test_variational_acc.npy", np.array(deepcopy(test_variational_accs)))
            np.save(self.log_path+ "_test_amortised_acc.npy", np.array(deepcopy(test_amortised_accs)))
            #save the weights:
            for (i,(layer, qlayer)) in enumerate(zip(self.layers, self.q_layers)):
                np.save(self.log_path + "_layer_"+str(i)+"_weights.npy",layer.weights)
                np.save(self.log_path + "_layer_"+str(i)+"_amortisation_weights.npy",qlayer.weights)

            #SAVE the results to the edinburgh computer from scratch space to main space
            subprocess.call(['rsync','--archive','--update','--compress','--progress',str(self.log_path) + "/",str(self.save_path)])
            tools.log("Rsynced files from: " + str(self.log_path) + "/ " + " to" + str(self.save_path))
            now = datetime.now()
            current_time = str(now.strftime("%H:%M:%S"))

            subprocess.call(['echo', f" TIME OF SAVE: {current_time}"])


    prediction_errors = np.array(prediction_errors)
    amortised_prediction_errors = np.array(amortised_prediction_errors)
    prediction_errors = torch.mean(torch.from_numpy(prediction_errors),dim=1).numpy()
    amortised_prediction_errors = torch.mean(torch.from_numpy(amortised_prediction_errors),dim=1).numpy()
    np.save(self.log_path + "_variational_acc.npy", np.array(variational_accs))
    np.save(self.log_path + "_amortised_acc.npy", np.array(amortised_accs))
    np.save(self.log_path + "_test_variational_acc.npy", np.array(test_variational_accs))
    np.save(self.log_path+ "_test_amortised_acc.npy", np.array(test_amortised_accs))
    #save the weights:
    for (i,(layer, qlayer)) in enumerate(zip(self.layers, self.q_layers)):
        np.save(self.log_path + "_layer_"+str(i)+"_weights.npy",layer.weights)
        np.save(self.log_path + "_layer_"+str(i)+"_amortisation_weights.npy",qlayer.weights)

    #SAVE the results to the edinburgh computer from scratch space to main space
    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(self.log_path) + "/",str(self.save_path)])
    tools.log("Rsynced files from: " + str(self.log_path) + "/ " + " to" + str(self.save_path))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))

    subprocess.call(['echo', f" TIME OF SAVE: {current_time}"])


def run_amortised(args):
    layer_sizes = [784, 300, 100, 10]
    n_layers = len(layer_sizes)

    train_set = torchvision.datasets.MNIST("MNIST_train", download=True, train=True)
    test_set = torchvision.datasets.MNIST("MNIST_test", download=True, train=False)
    if args.num_batches == -1:
        args.num_batches = len(train_set)// args.batch_size
    print("Num Batches",args.num_batches)
    img_list = [np.array([np.array(train_set[(n * args.batch_size) + i][0]).reshape([784, 1]) / 255.0 for i in range(args.batch_size)]).T.reshape([784, args.batch_size]) for n in range(args.num_batches)]
    label_list = [np.array([onehot(train_set[(n * args.batch_size) + i][1]) for i in range(args.batch_size)]).T for n in range(args.num_batches)]
    test_img_list = [np.array([np.array(test_set[(n * args.batch_size) + i][0]).reshape([784, 1]) / 255.0 for i in range(args.batch_size)]).T.reshape([784, args.batch_size]) for n in range(args.num_test_batches)]
    test_label_list = [np.array([onehot(test_set[(n * args.batch_size) + i][1]) for i in range(args.batch_size)]).T for n in range(args.num_test_batches)]
    pred_net = AmortisedPredictiveCodingNetwork(
        layer_sizes,
        tanh,
        tanhderiv,
        tanh,
        tanhderiv,
        args,
    )
    pred_net.train(img_list, label_list,test_img_list, test_label_list)

if __name__ == "__main__":

    def boolcheck(x):
        return str(x).lower() in ["true", "1", "yes"]

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--use_backward_nonlinearity",type=boolcheck, default="true")
    parser.add_argument("--weight_symmetry",type=boolcheck,default="true")
    parser.add_argument("--dropout_prob",type=int, default=-1)
    parser.add_argument("--backward_weight_update",type=boolcheck,default="true")
    parser.add_argument("--sign_concordance_prob",type=int, default=-1)
    parser.add_argument("--batch_size",type=int, default=10)
    parser.add_argument("--save_every",type=int, default=10)
    parser.add_argument("--num_batches",type=int,default=10)
    parser.add_argument("--num_test_batches",type=int, default=20)
    parser.add_argument("--n_epochs", type=int, default=101)
    parser.add_argument("--learning_rate",type=float, default=0.01)
    parser.add_argument("--amortised_learning_rate",type=float, default=0.001)
    parser.add_argument("--n_inference_steps_train",type=int, default=100)
    parser.add_argument("--n_inference_steps_test",type=int, default=1000)
    parser.add_argument("--inference_threshold",type=float, default=0.5)

    args =parser.parse_args()
    if args.dropout_prob == -1:
        args.dropout_prob = None
    if args.sign_concordance_prob == -1:
        args.sign_concordance_prob = None
    run_amortised(args)
