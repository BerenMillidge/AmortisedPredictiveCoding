import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from functions import *
from models import *
import sys
from copy import deepcopy
import os
import time
import subprocess
from datetime import datetime
import argparse

class PredictiveCodingLayer(object):
  def __init__(self,input_size, output_size, batch_size, fn,fn_deriv,learning_rate,dropout_prob = None,use_backward_nonlinearity=True):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.learning_rate = learning_rate
    self.dropout_prob = dropout_prob
    self.use_backward_nonlinearity = use_backward_nonlinearity
    self.weights = np.random.normal(0,0.1,[input_size, output_size])
    #self.backward_weights = same_sign(self.weights.T, np.random.normal(0,0.1,[input_size,output_size]).T)
    #self.backward_weights = np.random.normal(0,0.1,[input_size, output_size]).T
    self.backward_weights = self.weights.T
    # do dropout if required
    if self.dropout_prob is not None:
        self.weight_mask = dropout_mask(self.weights, self.dropout_prob)
        self.backward_weight_mask = dropout_mask(self.backward_weights,self.dropout_prob)
        self.weights *= self.weight_mask
        self.backward_weights *= self.backward_weight_mask
    print(input_size, batch_size)
    self.mu = np.random.normal(0,1,[output_size,batch_size])
    #a = np.random.normal(0,1,[10,10])
    #print(self.cosine_similarity(a,a)
    #bib


  def compute_mu_update_angles(self,pe,pe_below):
      if self.use_backward_nonlinearity:
          backward_weight_update =  np.dot(self.backward_weights, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))
          without_backward_weight_update = np.dot(self.weights.T, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))
          return self.cosine_similarity(backward_weight_update, without_backward_weight_update)
      else:
          backward_weight_update =  np.dot(self.backward_weights, pe_below)
          without_backward_weight_update = np.dot(self.weights.T, pe_below)
          return self.cosine_similarity(backward_weight_update, without_backward_weight_update)


  def cosine_similarity(self,w1,w2):
    cos = 0
    for i in range(10):
        cos += np.arccos(np.dot(w1[i,:].T,w2[i,:]) / (np.linalg.norm(w1) * np.linalg.norm(w2)))
    print(cos/10)
    return cos/10

  def update_mu(self, pe, pe_below):
    if self.use_backward_nonlinearity:
        #print("USING BACKWARD NONLINEARIT")
        self.mu+= self.learning_rate * (-pe + (np.dot(self.backward_weights, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))))
    else:
        #print("RUNNING WITHOUT BACKWAWRD NONLINEARITIES MU")
        self.mu+= self.learning_rate * (-pe + (np.dot(self.backward_weights, pe_below)))

  def step(self,pe_below,pred,use_top_down_pe=True):
    if use_top_down_pe:
      pe = self.mu - pred
    else:
      pe = np.zeros_like(self.mu)
    self.mu+= self.learning_rate * (-pe + (np.dot(self.backwards_weights, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))))
    return pe, self.predict()

  def predict(self):
    return self.fn(np.dot(self.weights, self.mu))

  def update_weights(self,pe_below):
    if self.use_backward_nonlinearity:
        self.weights += self.learning_rate * (np.dot(pe_below * self.fn_deriv(np.dot(self.weights, self.mu)), self.mu.T))
    else:
        #print("RUNNING WITHOUT BACKWARDS NONLINEARITY")
        self.weights += self.learning_rate * (np.dot(pe_below, self.mu.T))
    #reapply dropuot mask
    if self.dropout_prob is not None:
        self.weights *= self.weight_mask


  def update_backward_weights(self, pe_below):
    if self.use_backward_nonlinearity:
        print("USING NONLINEARITY")
        self.backward_weights += self.learning_rate * np.dot(self.mu,(pe_below * self.fn_deriv(np.dot(self.weights, self.mu))).T)
    else:
        #print("NO NONLINEARITY")
        self.backward_weights += self.learning_rate * np.dot(self.mu,pe_below.T)
    #reapply droput_mask
    if self.dropout_prob is not None:
        self.backward_weights *= self.backward_weight_mask


class PredictiveCodingNetwork(object):

  def __init__(self, layer_sizes,batch_size,learning_rate, f,df,n_inference_steps,dropout_prob =None):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.n_inference_steps = n_inference_steps
    self.dropout_prob = dropout_prob
    self.layers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate,self.dropout_prob))
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
    weight_angles = [[] for i in range(self.L)]
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
    #for b in range(batch_size):
    #  plt.plot(pred_labels[:,b])
    #  plt.show()
    #  print(true_labels[:,b])


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
    self.qf = qf
    self.dqf = dqf
    self.weights = np.random.normal(0,0.1,[self.forward_size, self.backward_size])
    self.use_backward_nonlinearity = use_backward_nonlinearity

  def predict(self, state):
    return self.qf(np.dot(self.weights, state))


  def update_weights(self, amortised_prediction_errors, state):
    if self.use_backward_nonlinearity:
        #print("USING NONLINEARITY")
        self.weights += self.learning_rate * (np.dot(amortised_prediction_errors * self.dqf(np.dot(self.weights, state)),state.T))
    else:
        print("AMORTISED NO BACKWARDS NONLINEARITY")
        self.weights += self.learning_rate * (np.dot(amortised_prediction_errors,state.T))


class AmortisedPredictiveCodingNetwork(object):
  def __init__(self, layer_sizes,batch_size,learning_rate,amortised_learning_rate,n_inference_steps_train,n_inference_steps_test, f,df,qf,dqf,inference_threshold=0.1,use_backward_weights = True,use_backward_nonlinearity=True,compute_weight_angles=True):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.amortised_learning_rate = amortised_learning_rate
    self.n_inference_steps_train = n_inference_steps_train
    self.n_inference_steps_test = n_inference_steps_test
    self.f = f
    self.df = df
    self.qf = qf
    self.dqf = dqf
    self.inference_threshold = inference_threshold
    self.use_backward_weights = use_backward_weights
    self.use_backward_nonlinearity = use_backward_nonlinearity
    self.compute_weight_angles = compute_weight_angles
    if self.compute_weight_angles:
        self.weight_angles = [[] for i in range(len(self.layer_sizes))]
        self.average_weight_angles =[]
    print("INITIALED: ", self.use_backward_nonlinearity)
    self.layers = []
    self.q_layers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate,use_backward_nonlinearity=self.use_backward_nonlinearity))
    #initialize amortised networks
    for i in range(len(self.layer_sizes)-1):
      self.q_layers.append(AmortisationLayer(self.layer_sizes[i+1], self.layer_sizes[i], self.amortised_learning_rate,self.batch_size, self.qf,self.dqf,use_backward_nonlinearity=self.use_backward_nonlinearity))
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

  def cosine_similarity(self,w1,w2):
    return np.arccos(np.sum(np.dot(w1.T,w2)) / (np.linalg.norm(w1) * np.linalg.norm(w2)))

  def forward_pass(self, img_batch, label_batch, test=False):
    # reset model
    n_inference_steps = self.n_inference_steps_test if test else self.n_inference_steps_train
    self.reset_mus()
    prediction_errors = [[] for i in range(self.n_layers)]
    if not test:
        if self.compute_weight_angles:
            weight_angles = [[] for i in range(self.L)]

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
                if self.compute_weight_angles and not test:
                    weight_angles[i].append(self.layers[i].compute_mu_update_angles(self.prediction_errors[i+1], self.prediction_errors[i]))
                self.layers[i].update_mu(self.prediction_errors[i+1], self.prediction_errors[i])
            else:
                if self.compute_weight_angles and not test:
                    weight_angles[i].append(self.layers[i].compute_mu_update_angles(self.prediction_errors[i+1], self.prediction_errors[i]))
                self.layers[i].update_mu(np.zeros_like(self.layers[i].mu), self.prediction_errors[i])
            self.predictions[i] = self.layers[i].predict()
            #set reset the final layer mus to the batch label
        if not test:
            self.layers[-1].mu = deepcopy(label_batch)
            self.predictions[-1] = deepcopy(self.layers[-1].mu)


        F = np.sum(np.array([np.mean(np.square(pe)) for pe in self.prediction_errors]))
        if F <= self.inference_threshold:
            break
    if not test:
        if self.compute_weight_angles:
            average_weight_angles = []
            for l in range(self.L):
                self.weight_angles[l].append(np.array(weight_angles[l]))
                #plt.plot(weight_angles[l])
                #plt.show()
                average_weight_angles.append(np.mean(np.array(weight_angles[l])))

            self.average_weight_angles.append(np.mean(np.array(average_weight_angles)))
    pred_imgs = deepcopy(self.predictions[0])
    pred_labels = deepcopy(self.layers[-1].mu)
    return pred_imgs, pred_labels

  def train_batch(self, img_batch, label_batch):
    self.forward_pass(img_batch, label_batch, test=False)
    #where am I going to add weight updates. It IS here somewhere... but where!
    for l in range(self.L):
        self.layers[l].update_weights(self.prediction_errors[l])
        if self.use_backward_weights:
            self.layers[l].update_backward_weights(self.prediction_errors[l])
        if self.compute_weight_angles:
            self.weight_angles[l].append(self.cosine_similarity(self.layers[l].weights.T, self.layers[l].backward_weights))

        self.amortised_prediction_errors[l] = self.layers[l].mu - self.amortised_predictions[l+1]
        #if l == 0:
        #    self.q_layers[l].update_weights(self.amortised_prediction_errors[l], self.amortised_predictions[0])
        #else:
        #    self.q_layers[l].update_weights(self.amortised_prediction_errors[l], self.layers[l-1].mu)

    return self.prediction_errors, self.predictions,self.amortised_predictions, self.amortised_prediction_errors


  def test(self, img_batch, label_batch):
    pred_imgs, pred_labels  = self.forward_pass(img_batch, label_batch,test=True)
    return pred_imgs, pred_labels

  def amortised_test(self, img_batch):
    self.amortisation_pass(img_batch,initialize=True)
    return self.amortised_predictions[-1]

  def plot_batch_results(self,pred_labels,amortised_labels, true_labels):
    batch_size = pred_labels.shape[1]
    #for b in range(batch_size):
    #  print("True labels: ", true_labels[:,b])
    #  print("Variational predictions: ")
    #  plt.plot(pred_labels[:,b])
    #  plt.show()
    #  print("Amortised Predictions: ",)
    #  plt.plot(amortised_labels[:,b])
    #  plt.show()

  def accuracy(self,pred_labels, true_labels):
      #print("IN accuracy", pred_labels, true_labels)
      correct = 0
      batch_size = pred_labels.shape[1]
      for b in range(batch_size):
        if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
          correct +=1
      return correct / batch_size


  def train(self, imglist, labellist,test_img_list, test_label_list, n_epochs,log_path, save_path):
    prediction_errors = []
    amortised_prediction_errors = []
    variational_accs = []
    amortised_accs = []
    test_variational_accs = []
    test_amortised_accs = []
    for n in range(n_epochs):
        print("Epoch ", n)
        batch_pes = []
        batch_qpes = []
        for (img_batch,label_batch) in zip(imglist, labellist):
            pes, preds,qpreds, qpes = self.train_batch(img_batch,label_batch)
            batch_pes.append(np.array([np.sum(pe) for pe in pes]))
            batch_qpes.append(np.array([np.sum(qpe) for qpe in qpes]))

        prediction_errors.append(np.array(batch_pes))
        amortised_prediction_errors.append(np.array(batch_qpes))

        if n % 1 == 0:
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
            #if self.compute_weight_angles:
            #    #print(self.weight_angles)
            #    plt.plot(self.average_weight_angles)
            #    plt.show()
            print("TEST ACCURACIES")
            tot_acc = 0
            """q_acc = 0
            for (test_img_batch, test_label_batch) in zip(test_img_list, test_label_list):
                pred_imgs, pred_labels = self.test(test_img_batch, test_label_batch)
                tot_acc += accuracy(pred_labels, test_label_batch)
                pred_qlabels = self.amortised_test(test_img_batch)
                q_acc += accuracy(pred_qlabels, test_label_batch)
            print(f"Test Variational Accuracy: {tot_acc/len(test_img_list)}")
            print(f"Test Amortised Accuracy: {q_acc / len(test_img_list)}")
            test_variational_accs.append(tot_acc/len(test_img_list))
            test_amortised_accs.append(q_acc / len(test_img_list))
            np.save(save_name + "_variational_acc.npy", np.array(deepcopy(variational_accs)))
            np.save(save_name + "_amortised_acc.npy", np.array(deepcopy(amortised_accs)))
            np.save(save_name + "_test_variational_acc.npy", np.array(deepcopy(test_variational_accs)))
            np.save(save_name+ "_test_amortised_acc.npy", np.array(deepcopy(test_amortised_accs)))"""
            np.save(log_path + "_variational_acc.npy", np.array(variational_accs))
            np.save(log_path + "_amortised_acc.npy", np.array(amortised_accs))
            np.save(log_path + "_test_variational_acc.npy", np.array(test_variational_accs))
            np.save(log_path+ "_test_amortised_acc.npy", np.array(test_amortised_accs))
            subprocess.call(['rsync','--archive','--update','--compress','--progress',str(log_path) +"/",str(save_path)])
            print("Rsynced files from: " + str(log_path) + "/ " + " to" + str(save_path))
            now = datetime.now()
            current_time = str(now.strftime("%H:%M:%S"))
            #save the weights:
            #for (i,(layer, qlayer)) in enumerate(zip(self.layers, self.q_layers)):
            #    np.save(save_name + "_layer_"+str(i)+"_weights.npy",layer.weights)
            #    np.save(save_name + "_layer_"+str(i)+"_amortisation_weights.npy",qlayer.weights)


    prediction_errors = np.array(prediction_errors)
    amortised_prediction_errors = np.array(amortised_prediction_errors)
    prediction_errors = torch.mean(torch.from_numpy(prediction_errors),dim=1).numpy()
    amortised_prediction_errors = torch.mean(torch.from_numpy(amortised_prediction_errors),dim=1).numpy()

    np.save(log_path + "_variational_acc.npy", np.array(variational_accs))
    np.save(log_path + "_amortised_acc.npy", np.array(amortised_accs))
    np.save(log_path + "_test_variational_acc.npy", np.array(test_variational_accs))
    np.save(log_path+ "_test_amortised_acc.npy", np.array(test_amortised_accs))
    #save the weights:
    #for (i,(layer, qlayer)) in enumerate(zip(self.layers, self.q_layers)):
    #    np.save(save_name + "_layer_"+str(i)+"_weights.npy",layer.weights)
    #    np.save(save_name + "_layer_"+str(i)+"_amortisation_weights.npy",qlayer.weights)
    #SAVE the results to the edinburgh computer from scratch space to main space
    subprocess.call(['rsync','--archive','--update','--compress','--progress',str(log_path) +"/",str(save_path)])
    print("Rsynced files from: " + str(log_path) + "/ " + " to" + str(save_path))
    now = datetime.now()
    current_time = str(now.strftime("%H:%M:%S"))

    #subprocess.call(['echo', f" TIME OF SAVE: {current_time}"])
    """print("Accuracy plots")
    plt.title("Variational Accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Proportion correct")
    plt.plot(variational_accs)
    plt.show()
    plt.title("Amortised Accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Proportion correct")
    plt.plot(amortised_accs)
    plt.show()
    print("Test Accuracy plots")
    plt.title("Variational Accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Proportion correct")
    plt.plot(test_variational_accs)
    plt.show()
    plt.title("Amortised Accuracy")
    plt.xlabel("Epoch number")
    plt.ylabel("Proportion correct")
    plt.plot(test_amortised_accs)
    plt.show()"""


def run_amortised(log_path, save_path,use_backward_weights,use_backward_nonlinearity,compute_weight_angles):
    batch_size = 10
    num_batches = 10
    num_test_batches = 20
    n_inference_steps_train = 100
    n_inference_steps_test = 1000
    learning_rate = 0.001
    amortised_learning_rate = 0.01
    layer_sizes = [784, 300, 100, 10]
    n_layers = len(layer_sizes)
    n_epochs = 101
    inference_thresh = 0.1

    train_set = torchvision.datasets.MNIST("MNIST_train", download=True, train=True)
    test_set = torchvision.datasets.MNIST("MNIST_test", download=True, train=False)
    #num_batches = len(train_set)// batch_size
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
        f=linear,
        df=linearderiv,
        qf=linear,
        dqf=linearderiv,
        inference_threshold=inference_thresh,
        use_backward_weights = use_backward_weights,
        use_backward_nonlinearity = use_backward_nonlinearity,
        compute_weight_angles = compute_weight_angles,
    )
    pred_net.train(img_list, label_list,test_img_list, test_label_list, n_epochs,log_path, save_path)

if __name__ == "__main__":
    log_path = str(sys.argv[1])
    save_path = str(sys.argv[2])
    use_backward_weights = True
    use_backward_nonlinearity = True
    if len(sys.argv) > 3:
        if sys.argv[3] == "False":
            use_backward_weights = False
    if len(sys.argv) > 4:
        if sys.argv[4] == "False":
            use_backward_nonlinearity = False
    compute_weight_angles = False
    run_amortised(log_path, save_path,use_backward_weights,use_backward_nonlinearity,compute_weight_angles)
