import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch

class PredictiveCodingLayer(object):
  def __init__(self,input_size, output_size, batch_size, fn,fn_deriv,learning_rate):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.learning_rate = learning_rate
    self.weights = np.random.normal(0,0.1,[input_size, output_size])
    print(input_size, batch_size)
    self.mu = np.random.normal(0,1,[output_size,batch_size])

  def _update_mu(self, pe, pe_below):
    self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))))

  def step(self,pe_below,pred,use_top_down_pe=True):
    if use_top_down_pe:
      pe = self.mu - pred
    else:
      pe = np.zeros_like(self.mu)
    self.mu+= self.learning_rate * (-pe + (np.dot(self.weights.T, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))))
    return pe, self.predict()

  def predict(self):
    return self.fn(np.dot(self.weights, self.mu))

  def update_weights(self,pe_below):
    self.weights += self.learning_rate * (np.dot(pe_below * self.fn_deriv(np.dot(self.weights, self.mu)), self.mu.T))

class PredictiveCodingNetwork(object):

  def __init__(self, layer_sizes,batch_size,learning_rate, f,df,n_inference_steps):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.n_inference_steps = n_inference_steps
    self.layers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate))
    self.predictions = [[]for i in range(len(self.layer_sizes))]
    self.prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.L = len(self.layers)

  def _reset_mus(self):
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

  def accuracy(self,pred_labels, true_labels):
    correct = 0
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
        correct +=1
    return correct / batch_size

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
  def __init__(self, forward_size, backward_size, learning_rate, batch_size, qf, dqf):
    self.forward_size = forward_size
    self.backward_size = backward_size
    self.learning_rate = learning_rate
    self.batch_szie = batch_size
    self.qf = qf
    self.dqf = dqf
    self.weights = np.random.normal(0,0.1,[self.forward_size, self.backward_size])

  def predict(self, state):
    return self.qf(np.dot(self.weights, state))

  def update_weights(self, amortised_prediction_errors, state):
    self.weights += self.learning_rate * (np.dot(amortised_prediction_errors * self.dqf(np.dot(self.weights, state)),state.T))


class AmortisedPredictiveCodingNetwork(object):
  def __init__(self, layer_sizes,batch_size,learning_rate,amortised_learning_rate, f,df,qf,dqf,n_inference_steps):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.amortised_learning_rate = amortised_learning_rate
    self.f = f
    self.df = df
    self.qf = qf
    self.dqf = dqf
    self.n_inference_steps = n_inference_steps
    self.layers = []
    self.qlayers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate))
    #initialize amortised networks
    for i in range(len(self.layer_sizes)-1):
      self.qlayers.append(AmortisationLayer(self.layer_sizes[i+1], self.layer_sizes[i], self.amortised_learning_rate,self.batch_size, self.qf,self.dqf))
    self.predictions = [[]for i in range(len(self.layer_sizes))]
    self.prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.amortised_predictions = [[] for i in range(len(self.layer_sizes))]
    self.amortised_prediction_errors = [[] for i in range(len(self.layer_sizes))]
    self.L = len(self.layers)

  def _reset_mus(self):
    for layer in self.layers:
        layer.mu = np.random.normal(0,1,[layer.output_size,layer.batch_size])

  def infer(self, img_batch,label_batch):
    self._reset_mus()
    prediction_errors = [[] for i in range(self.L)]
    self.layers[-1].mu = deepcopy(label_batch)
    self.amortised_predictions[0] = deepcopy(img_batch)
    for l in range(self.L):
      self.amortised_predictions[l+1] = self.qlayers[l].predict(self.amortised_predictions[l])
      self.layers[l].mu = deepcopy(self.amortised_predictions[l+1])
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
      self.amortised_prediction_errors[l] = self.layers[l].mu - self.amortised_predictions[l+1]
      if l == 0:
        self.qlayers[l].update_weights(self.amortised_prediction_errors[l], self.amortised_predictions[0])
      else:
        self.qlayers[l].update_weights(self.amortised_prediction_errors[l],self.layers[l-1].mu)

    return self.prediction_errors, self.predictions,self.amortised_predictions,self.amortised_prediction_errors

  def test(self, img_batch, label_batch):
    self._reset_mus()
    prediction_errors = [[] for i in range(self.L)]
    self.amortised_predictions[0] = deepcopy(img_batch)
    for l in range(self.L):
      self.amortised_predictions[l+1] = self.qlayers[l].predict(self.amortised_predictions[l])
      self.layers[l].mu = deepcopy(self.amortised_predictions[l+1])
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

  def amortised_test(self, img_batch):
    self.amortised_predictions[0] = deepcopy(img_batch)
    for l in range(self.L):
      self.amortised_predictions[l+1] = self.qlayers[l].predict(self.amortised_predictions[l])
    pred_qlabels = self.amortised_predictions[-1]
    return pred_qlabels


  def accuracy(self,pred_labels, true_labels):
    correct = 0
    batch_size = pred_labels.shape[1]
    for b in range(batch_size):
      if np.argmax(pred_labels[:,b]) == np.argmax(true_labels[:,b]):
        correct +=1
    return correct / batch_size

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


  def train(self, imglist, labellist, n_epochs):
    prediction_errors = []
    amortised_prediction_errors = []
    variational_accs = []
    amortised_accs = []
    for n in range(n_epochs):
        print("Epoch ", n)
        batch_pes = []
        batch_qpes = []
        for (img_batch,label_batch) in zip(imglist, labellist):
            pes, preds,qpreds, qpes = self.infer(img_batch,label_batch)
            batch_pes.append(np.array([np.sum(pe) for pe in pes]))
            batch_qpes.append(np.array([np.sum(qpe) for qpe in qpes]))

        prediction_errors.append(np.array(batch_pes))
        amortised_prediction_errors.append(np.array(batch_qpes))

        if n % 10 == 0:
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


    prediction_errors = np.array(prediction_errors)
    amortised_prediction_errors = np.array(amortised_prediction_errors)
    print("Prediction error shapes")
    print(prediction_errors.shape)
    print(amortised_prediction_errors.shape)
    prediction_errors = torch.mean(torch.from_numpy(prediction_errors),dim=1).numpy()
    amortised_prediction_errors = torch.mean(torch.from_numpy(amortised_prediction_errors),dim=1).numpy()
    print(prediction_errors.shape)
    print(amortised_prediction_errors.shape)
    for i in range(self.L):
        plt.title("Average Variational Prediction Errors Layer " + str(i))
        plt.xlabel("Epoch")
        plt.ylabel("Prediction Errors")
        plt.plot(prediction_errors[:,i])
        plt.show()
        plt.title("Average Amortised Prediction Errors Layer " + str(i))
        plt.xlabel("Epoch")
        plt.ylabel("Prediction Errors")
        plt.plot(amortised_prediction_errors[:,i])
        plt.show()
    #pred_imgs, pred_labels = self.test(imglist[0],labellist[0])
    #pred_qlabels = self.amortised_test(imglist[0])
    #self.plot_batch_results(pred_labels, pred_qlabels, labellist[0])
    print("Accuracy plots")
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
