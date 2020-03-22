# code to check the performance of the alg for each sign concordance PROBABILITY
#should hopefully be straightforwardly implemented and tested

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import torch
from functions import *
import torchvision

class PredictiveCodingLayer(object):
  def __init__(self,input_size, output_size, batch_size, fn,fn_deriv,learning_rate,dropout_prob = None,sign_concordance_prob=1):
    self.input_size = input_size
    self.output_size = output_size
    self.batch_size = batch_size
    self.fn = fn
    self.fn_deriv = fn_deriv
    self.learning_rate = learning_rate
    self.dropout_prob = dropout_prob
    self.sign_concordance_prob = sign_concordance_prob
    self.weights = np.random.normal(0,0.1,[input_size, output_size])
    self.backward_weights = sign_concordance(self.weights.T, np.random.normal(0,0.1,[input_size,output_size]).T,self.sign_concordance_prob)
    #self.backward_weights = deepcopy(self.weights.T)
    # do dropout if required
    if self.dropout_prob is not None:
        self.weight_mask = dropout_mask(self.weights, self.dropout_prob)
        self.backward_weight_mask = dropout_mask(self.backward_weights,self.dropout_prob)
        self.weights *= self.weight_mask
        self.backward_weights *= self.backward_weight_mask
    print(input_size, batch_size)
    self.mu = np.random.normal(0,1,[output_size,batch_size])

  def update_mu(self, pe, pe_below):
    self.mu+= self.learning_rate * (-pe + (np.dot(self.backward_weights, pe_below *self.fn_deriv(np.dot(self.weights, self.mu)))))

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
    self.weights += self.learning_rate * np.dot(pe_below ,self.mu.T)
    #reapply dropuot mask
    if self.dropout_prob is not None:
        self.weights *= self.weight_mask


  def update_backward_weights(self, pe_below):
    self.backward_weights += self.learning_rate * np.dot(self.mu,pe_below.T)
    #reapply droput_mask
    if self.dropout_prob is not None:
        self.backward_weights *= self.backward_weight_mask


class PredictiveCodingNetwork(object):

  def __init__(self, layer_sizes,batch_size,learning_rate, f,df,n_inference_steps,dropout_prob =None,sign_concordance_prob=1):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.f = f
    self.df = df
    self.n_inference_steps = n_inference_steps
    self.dropout_prob = dropout_prob
    self.sign_concordance_prob = sign_concordance_prob
    self.layers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate,self.dropout_prob,self.sign_concordance_prob))
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
    self.weights += self.learning_rate * (np.dot(amortised_prediction_errors,state.T))


class AmortisedPredictiveCodingNetwork(object):
  def __init__(self, layer_sizes,batch_size,learning_rate,amortised_learning_rate, f,df,qf,dqf,n_inference_steps,dropout_prob = None,sign_concordance_prob = 1):
    self.layer_sizes = layer_sizes
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.amortised_learning_rate = amortised_learning_rate
    self.f = f
    self.df = df
    self.qf = qf
    self.dqf = dqf
    self.n_inference_steps = n_inference_steps
    self.dropout_prob = dropout_prob
    self.sign_concordance_prob = sign_concordance_prob
    self.layers = []
    self.q_layers = []
    for i in range(len(self.layer_sizes)-1):
      self.layers.append(PredictiveCodingLayer(self.layer_sizes[i], self.layer_sizes[i+1], self.batch_size, self.f,self.df,self.learning_rate,self.dropout_prob,self.sign_concordance_prob))
    #initialize amortised networks
    for i in range(len(self.layer_sizes)-1):
      self.q_layers.append(AmortisationLayer(self.layer_sizes[i+1], self.layer_sizes[i], self.amortised_learning_rate,self.batch_size, self.qf,self.dqf))
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
    n_inference_steps = self.n_inference_steps * 10 if test else self.n_inference_steps
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

    pred_imgs = deepcopy(self.predictions[0])
    pred_labels = deepcopy(self.layers[-1].mu)
    return pred_imgs, pred_labels

  def train_batch(self, img_batch, label_batch):
    self.forward_pass(img_batch, label_batch, test=False)
    for l in range(self.L):
        self.layers[l].update_weights(self.prediction_errors[l])
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
            pes, preds,qpreds, qpes = self.train_batch(img_batch,label_batch)
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
    np.save("sign_concordance_"+str(self.sign_concordance_prob).split(".")[1] + "_variational_accs.npy",np.array(variational_accs))
    np.save("sign_concordance_"+str(self.sign_concordance_prob).split(".")[1] + "amortised_accs.npy",np.array(amortised_accs))
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

def run_amortised(trainset, valset,sign_concordance_prob):
    fn = tanh
    fn_deriv = tanhderiv
    qf = tanh
    qf_deriv = tanhderiv
    batch_size=10
    num_batches = 10
    n_inference_steps = 100
    learning_rate= 0.01
    amortised_learning_rate = 0.001
    layer_sizes = [784,300,100,10]
    L = len(layer_sizes)
    n_epochs = 101
    dropout_prob = None
    imglist = [np.array([np.array(trainset[(n * batch_size) + i][0]).reshape([784,1]) / 255. for i in range(batch_size)]).T.reshape([784,batch_size]) for n in range(num_batches)]
    labellist = [np.array([onehot(trainset[(n * batch_size) + i][1]) for i in range(batch_size)]).T for n in range(num_batches)]
    prednet = AmortisedPredictiveCodingNetwork(layer_sizes, batch_size,learning_rate,amortised_learning_rate, fn, fn_deriv,qf, qf_deriv,n_inference_steps,dropout_prob,sign_concordance_prob)
    prediction_errors = prednet.train(imglist, labellist, n_epochs)

if __name__ == "__main__":
    trainset = torchvision.datasets.MNIST('MNIST_train', download=True, train=True)
    valset = torchvision.datasets.MNIST('MNIST_test', download=True, train=False)
    concordance_probs = [0.5,0.6,0.7,0.8,0.9,1]
    for concordance_prob in concordance_probs:
        run_amortised(trainset, valset,concordance_prob)
